#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <stdint.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include "../../../common/rodinia_verify.h"

#define int2 int32_t
#define ulong4 uint32_t
#define uint4 uint32_t
#include "mummergpu.h"

char * OPT_reffilename = NULL;
char * OPT_qryfilename = NULL;
char * OPT_dotfilename = NULL;
char * OPT_texfilename = NULL;
int    OPT_num_reference_pages = 1;
char * OPT_stats_file = NULL;
char * OPT_save_reference = NULL;
char * OPT_verify_reference = NULL;
char * OPT_output_file = NULL;
bool OPT_suppress_stdout_output = false;

// MUMmer options
int  OPT_match_length = 20;
bool OPT_reverse = false;
bool OPT_forwardreverse = false;
bool OPT_forwardcoordinates = false;
bool OPT_showQueryLength = false;
bool OPT_maxmatch = false;
bool OPT_on_cpu = false;
bool OPT_stream_queries = false;

struct InputDigest
{
   unsigned long long hash;
   unsigned long long bytes;
};

struct MummerGpuReference
{
   char ref_name[512];
   char query_name[512];
   unsigned long long ref_hash;
   unsigned long long ref_bytes;
   unsigned long long query_hash;
   unsigned long long query_bytes;
   int match_length;
   int reverse;
   int forwardreverse;
   int forwardcoordinates;
   int show_query_length;
   int maxmatch;
   int treetex;
   int reorder_tree;
   int renumber_tree;
   int qrytex;
   int coalesced_queries;
   int reftex;
   int reorder_ref;
   unsigned long long output_hash;
   unsigned long long output_bytes;
   unsigned long long output_lines;
   unsigned long long output_headers;
};

static const unsigned long long FNV_OFFSET_BASIS = 1469598103934665603ULL;
static const unsigned long long FNV_PRIME = 1099511628211ULL;
static const char* MUMMERGPU_REFERENCE_MAGIC = "GPIDL_RODINIA_MUMMERGPU_REFERENCE";

void printHelp()
{
   fprintf(stderr, 
		   "Align a set of query strings to a reference sequence using the GPU.\n"
		   "\n"
		   "Usage: mummergpu [options] reference.fa query.fa\n"
		   "\n"
		   "Options\n"
		   "  -h Print this help\n"
		   "  -d file.dot Output suffix tree in dot format\n"
		   "  -t file.tex Output suffix tree texture\n"
		   "  -C             Compute the matches using the CPU instead of the GPU\n"
		   "  -s <file>      write timing and memory stats to <file> \n"
           "  --save-reference <file>   save a deterministic output summary\n"
           "  --verify-reference <file> verify a deterministic output summary\n"
           "  --output-file <file>      also write alignments to <file>\n"
           "  --suppress-stdout-output  write alignments only to --output-file\n"
           "\n"
           "  -l <matchlen>  minimal match length to report [Default: 20]\n"
           "  -b             compute forward and reverse complement matches\n"
           "  -r             only compute reverse complement matches\n"
           "  -c             report the query-position of a reverse complement match\n"
           "                 relative to the original query sequence\n"
           "  -L             show the length of the query sequences on the header line\n"
	  );

   exit(0);
}

void printUsage()
{
   fprintf(stderr, "Usage: mummergpu [options] reference.fa query.fa\n");
   exit(0);
}

static const char* basenameOf(const char* path)
{
   const char* slash = strrchr(path, '/');
   return slash ? slash + 1 : path;
}

static int computeInputDigest(const char* path, InputDigest* digest)
{
   FILE* file = fopen(path, "rb");
   if (!file)
   {
      fprintf(stderr, "Cannot open input for digest: %s\n", path);
      return 1;
   }

   digest->hash = FNV_OFFSET_BASIS;
   digest->bytes = 0;
   unsigned char buffer[65536];
   while (true)
   {
      size_t count = fread(buffer, 1, sizeof(buffer), file);
      for (size_t index = 0; index < count; ++index)
      {
         digest->hash ^= buffer[index];
         digest->hash *= FNV_PRIME;
      }
      digest->bytes += count;
      if (count != sizeof(buffer))
      {
         if (ferror(file))
         {
            fprintf(stderr, "Failed reading input for digest: %s\n", path);
            fclose(file);
            return 1;
         }
         break;
      }
   }

   fclose(file);
   return 0;
}

static int scanString(FILE* file, const char* expected_label, char* value, size_t value_size)
{
   char label[128];
   if (fscanf(file, "%127s %511s", label, value) != 2 || strcmp(label, expected_label) != 0)
   {
      fprintf(stderr, "Invalid mummergpu reference field: %s\n", expected_label);
      return 1;
   }
   value[value_size - 1] = 0;
   return 0;
}

static int scanULL(FILE* file, const char* expected_label, unsigned long long* value)
{
   char label[128];
   if (fscanf(file, "%127s %llx", label, value) != 2 || strcmp(label, expected_label) != 0)
   {
      fprintf(stderr, "Invalid mummergpu reference field: %s\n", expected_label);
      return 1;
   }
   return 0;
}

static int scanDecimalULL(FILE* file, const char* expected_label, unsigned long long* value)
{
   char label[128];
   if (fscanf(file, "%127s %llu", label, value) != 2 || strcmp(label, expected_label) != 0)
   {
      fprintf(stderr, "Invalid mummergpu reference field: %s\n", expected_label);
      return 1;
   }
   return 0;
}

static int scanInt(FILE* file, const char* expected_label, int* value)
{
   char label[128];
   if (fscanf(file, "%127s %d", label, value) != 2 || strcmp(label, expected_label) != 0)
   {
      fprintf(stderr, "Invalid mummergpu reference field: %s\n", expected_label);
      return 1;
   }
   return 0;
}

static void populateReference(
   MummerGpuReference* reference,
   const InputDigest* ref_digest,
   const InputDigest* query_digest,
   const MummerGpuOutputSummary* output)
{
   memset(reference, 0, sizeof(*reference));
   snprintf(reference->ref_name, sizeof(reference->ref_name), "%s", basenameOf(OPT_reffilename));
   snprintf(reference->query_name, sizeof(reference->query_name), "%s", basenameOf(OPT_qryfilename));
   reference->ref_hash = ref_digest->hash;
   reference->ref_bytes = ref_digest->bytes;
   reference->query_hash = query_digest->hash;
   reference->query_bytes = query_digest->bytes;
   reference->match_length = OPT_match_length;
   reference->reverse = OPT_reverse ? 1 : 0;
   reference->forwardreverse = OPT_forwardreverse ? 1 : 0;
   reference->forwardcoordinates = OPT_forwardcoordinates ? 1 : 0;
   reference->show_query_length = OPT_showQueryLength ? 1 : 0;
   reference->maxmatch = OPT_maxmatch ? 1 : 0;
   reference->treetex = TREETEX;
   reference->reorder_tree = REORDER_TREE;
   reference->renumber_tree = RENUMBER_TREE;
   reference->qrytex = QRYTEX;
   reference->coalesced_queries = COALESCED_QUERIES;
   reference->reftex = REFTEX;
   reference->reorder_ref = REORDER_REF;
   reference->output_hash = output->hash;
   reference->output_bytes = output->bytes;
   reference->output_lines = output->lines;
   reference->output_headers = output->headers;
}

static int saveReference(const char* path, const MummerGpuReference* reference)
{
   FILE* file = fopen(path, "w");
   if (!file)
   {
      fprintf(stderr, "Cannot open mummergpu reference for write: %s\n", path);
      return 1;
   }

   fprintf(file, "%s 1\n", MUMMERGPU_REFERENCE_MAGIC);
   fprintf(file, "ref_name %s\n", reference->ref_name);
   fprintf(file, "query_name %s\n", reference->query_name);
   fprintf(file, "ref_hash %016llx\n", reference->ref_hash);
   fprintf(file, "ref_bytes %llu\n", reference->ref_bytes);
   fprintf(file, "query_hash %016llx\n", reference->query_hash);
   fprintf(file, "query_bytes %llu\n", reference->query_bytes);
   fprintf(file, "match_length %d\n", reference->match_length);
   fprintf(file, "reverse %d\n", reference->reverse);
   fprintf(file, "forwardreverse %d\n", reference->forwardreverse);
   fprintf(file, "forwardcoordinates %d\n", reference->forwardcoordinates);
   fprintf(file, "show_query_length %d\n", reference->show_query_length);
   fprintf(file, "maxmatch %d\n", reference->maxmatch);
   fprintf(file, "treetex %d\n", reference->treetex);
   fprintf(file, "reorder_tree %d\n", reference->reorder_tree);
   fprintf(file, "renumber_tree %d\n", reference->renumber_tree);
   fprintf(file, "qrytex %d\n", reference->qrytex);
   fprintf(file, "coalesced_queries %d\n", reference->coalesced_queries);
   fprintf(file, "reftex %d\n", reference->reftex);
   fprintf(file, "reorder_ref %d\n", reference->reorder_ref);
   fprintf(file, "output_hash %016llx\n", reference->output_hash);
   fprintf(file, "output_bytes %llu\n", reference->output_bytes);
   fprintf(file, "output_lines %llu\n", reference->output_lines);
   fprintf(file, "output_headers %llu\n", reference->output_headers);

   if (fclose(file) != 0)
   {
      fprintf(stderr, "Failed to close mummergpu reference: %s\n", path);
      return 1;
   }

   printf("MUMmerGPU reference saved to '%s'\n", path);
   return 0;
}

static int loadReferenceFile(const char* path, MummerGpuReference* reference)
{
   FILE* file = fopen(path, "r");
   if (!file)
   {
      fprintf(stderr, "Cannot open mummergpu reference for read: %s\n", path);
      return 1;
   }

   char magic[128];
   int version = 0;
   if (fscanf(file, "%127s %d", magic, &version) != 2 ||
       strcmp(magic, MUMMERGPU_REFERENCE_MAGIC) != 0 ||
       version != 1)
   {
      fprintf(stderr, "Invalid mummergpu reference header\n");
      fclose(file);
      return 1;
   }

   int status =
      scanString(file, "ref_name", reference->ref_name, sizeof(reference->ref_name)) ||
      scanString(file, "query_name", reference->query_name, sizeof(reference->query_name)) ||
      scanULL(file, "ref_hash", &reference->ref_hash) ||
      scanDecimalULL(file, "ref_bytes", &reference->ref_bytes) ||
      scanULL(file, "query_hash", &reference->query_hash) ||
      scanDecimalULL(file, "query_bytes", &reference->query_bytes) ||
      scanInt(file, "match_length", &reference->match_length) ||
      scanInt(file, "reverse", &reference->reverse) ||
      scanInt(file, "forwardreverse", &reference->forwardreverse) ||
      scanInt(file, "forwardcoordinates", &reference->forwardcoordinates) ||
      scanInt(file, "show_query_length", &reference->show_query_length) ||
      scanInt(file, "maxmatch", &reference->maxmatch) ||
      scanInt(file, "treetex", &reference->treetex) ||
      scanInt(file, "reorder_tree", &reference->reorder_tree) ||
      scanInt(file, "renumber_tree", &reference->renumber_tree) ||
      scanInt(file, "qrytex", &reference->qrytex) ||
      scanInt(file, "coalesced_queries", &reference->coalesced_queries) ||
      scanInt(file, "reftex", &reference->reftex) ||
      scanInt(file, "reorder_ref", &reference->reorder_ref) ||
      scanULL(file, "output_hash", &reference->output_hash) ||
      scanDecimalULL(file, "output_bytes", &reference->output_bytes) ||
      scanDecimalULL(file, "output_lines", &reference->output_lines) ||
      scanDecimalULL(file, "output_headers", &reference->output_headers);

   fclose(file);
   return status;
}

static int compareULL(
   const char* field,
   unsigned long long actual,
   unsigned long long expected,
   bool hexadecimal)
{
   if (actual == expected)
   {
      return 0;
   }
   if (hexadecimal)
   {
      fprintf(stderr,
              "MUMmerGPU reference mismatch for %s: actual=%016llx expected=%016llx\n",
              field,
              actual,
              expected);
   }
   else
   {
      fprintf(stderr,
              "MUMmerGPU reference mismatch for %s: actual=%llu expected=%llu\n",
              field,
              actual,
              expected);
   }
   return 1;
}

static int compareInt(const char* field, int actual, int expected)
{
   if (actual == expected)
   {
      return 0;
   }
   fprintf(stderr,
           "MUMmerGPU reference mismatch for %s: actual=%d expected=%d\n",
           field,
           actual,
           expected);
   return 1;
}

static int compareString(const char* field, const char* actual, const char* expected)
{
   if (strcmp(actual, expected) == 0)
   {
      return 0;
   }
   fprintf(stderr,
           "MUMmerGPU reference mismatch for %s: actual=%s expected=%s\n",
           field,
           actual,
           expected);
   return 1;
}

static int verifyReference(const char* path, const MummerGpuReference* actual)
{
   MummerGpuReference expected;
   if (loadReferenceFile(path, &expected) != 0)
   {
      return 1;
   }

   int mismatches = 0;
   mismatches += compareString("ref_name", actual->ref_name, expected.ref_name);
   mismatches += compareString("query_name", actual->query_name, expected.query_name);
   mismatches += compareULL("ref_hash", actual->ref_hash, expected.ref_hash, true);
   mismatches += compareULL("ref_bytes", actual->ref_bytes, expected.ref_bytes, false);
   mismatches += compareULL("query_hash", actual->query_hash, expected.query_hash, true);
   mismatches += compareULL("query_bytes", actual->query_bytes, expected.query_bytes, false);
   mismatches += compareInt("match_length", actual->match_length, expected.match_length);
   mismatches += compareInt("reverse", actual->reverse, expected.reverse);
   mismatches += compareInt("forwardreverse", actual->forwardreverse, expected.forwardreverse);
   mismatches += compareInt("forwardcoordinates", actual->forwardcoordinates, expected.forwardcoordinates);
   mismatches += compareInt("show_query_length", actual->show_query_length, expected.show_query_length);
   mismatches += compareInt("maxmatch", actual->maxmatch, expected.maxmatch);
   mismatches += compareInt("treetex", actual->treetex, expected.treetex);
   mismatches += compareInt("reorder_tree", actual->reorder_tree, expected.reorder_tree);
   mismatches += compareInt("renumber_tree", actual->renumber_tree, expected.renumber_tree);
   mismatches += compareInt("qrytex", actual->qrytex, expected.qrytex);
   mismatches += compareInt("coalesced_queries", actual->coalesced_queries, expected.coalesced_queries);
   mismatches += compareInt("reftex", actual->reftex, expected.reftex);
   mismatches += compareInt("reorder_ref", actual->reorder_ref, expected.reorder_ref);
   mismatches += compareULL("output_hash", actual->output_hash, expected.output_hash, true);
   mismatches += compareULL("output_bytes", actual->output_bytes, expected.output_bytes, false);
   mismatches += compareULL("output_lines", actual->output_lines, expected.output_lines, false);
   mismatches += compareULL("output_headers", actual->output_headers, expected.output_headers, false);

   if (mismatches != 0)
   {
      fprintf(stderr, "MUMmerGPU reference verification failed with %d mismatch(es)\n", mismatches);
      return 1;
   }

   printf("MUMmerGPU reference verification matched '%s'\n", path);
   rodinia_print_pass("MUMmerGPU reference verification");
   return 0;
}


void ParseCommandLine(int argc, char ** argv)
{
   bool errflg = false;
   int ch;
   optarg = NULL;
   static const struct option long_options[] = {
      {"save-reference", required_argument, NULL, 1000},
      {"verify-reference", required_argument, NULL, 1001},
      {"output-file", required_argument, NULL, 1002},
      {"suppress-stdout-output", no_argument, NULL, 1003},
      {NULL, 0, NULL, 0}
   };

   while(!errflg && ((ch = getopt_long(argc, argv, "aCchql:d:t:s:brcLM", long_options, NULL)) != EOF))
   {
      switch  (ch)
	  {
		 case 'h': printHelp(); break;
		 case '?': fprintf(stderr, "Unknown option %c\n", optopt); errflg = true; break;
		 case 'd': OPT_dotfilename = optarg; break;
		 case 't': OPT_texfilename = optarg; break;
		 case 'C': OPT_on_cpu = true; break;
		 case 'l': OPT_match_length = atoi(optarg); break;
         case 'b': OPT_forwardreverse = true; break;
         case 'r': OPT_reverse = true; break;
		 case 's': OPT_stats_file = optarg; break;
         case 'c': OPT_forwardcoordinates = true; break;
         case 'L': OPT_showQueryLength = true; break;
         case 'M': OPT_maxmatch = true; break;
         case 1000: OPT_save_reference = optarg; break;
         case 1001: OPT_verify_reference = optarg; break;
         case 1002: OPT_output_file = optarg; break;
         case 1003: OPT_suppress_stdout_output = true; break;

		 default: errflg = true; break;
	  };
   }

   if ((optind != argc-2) || errflg) { printUsage(); }
   if (OPT_save_reference && OPT_verify_reference)
   {
      fprintf(stderr, "ERROR: Only one reference file mode may be specified\n");
      exit(1);
   }

   if (!OPT_maxmatch)
   {
     OPT_maxmatch = true;
   }

   if (OPT_reverse && OPT_forwardreverse)
   {
     fprintf(stderr, "ERROR: Reverse (-r) and Forward & Reverse (-b) specified\n");
     exit(1);
   }

   OPT_reffilename = argv[optind++];
   OPT_qryfilename = argv[optind++];
}

int main(int argc, char* argv[])
{
   ParseCommandLine(argc, argv);

   fprintf(stderr, "TWO_LEVEL_NODE_TREE is %d\n", TWO_LEVEL_NODE_TREE);
   fprintf(stderr, "TWO_LEVEL_CHILD_TREE is %d\n", TWO_LEVEL_CHILD_TREE);
   fprintf(stderr, "QRYTEX is %d\n", QRYTEX);
   fprintf(stderr, "COALESCED_QUERIES is %d\n", COALESCED_QUERIES);
   fprintf(stderr, "REFTEX is %d\n", REFTEX);
   fprintf(stderr, "REORDER_REF is %d\n", REORDER_REF);
   fprintf(stderr, "NODETEX is %d\n", NODETEX);
   fprintf(stderr, "CHILDTEX is %d\n", CHILDTEX);
   fprintf(stderr, "MERGETEX is %d\n", MERGETEX);
   fprintf(stderr, "REORDER_TREE is %d\n", REORDER_TREE);
	fprintf(stderr, "RENUMBER_TREE is %d\n", RENUMBER_TREE);

   int err = 0;
   InputDigest ref_digest;
   InputDigest query_digest;
   if (computeInputDigest(OPT_reffilename, &ref_digest) != 0 ||
       computeInputDigest(OPT_qryfilename, &query_digest) != 0)
   {
      exit(1);
   }

   resetMummerGpuOutputSummary();
   if (OPT_suppress_stdout_output && !OPT_output_file)
   {
      fprintf(stderr, "ERROR: --suppress-stdout-output requires --output-file\n");
      exit(1);
   }
   setMummerGpuSuppressStdout(OPT_suppress_stdout_output ? 1 : 0);
   if (setMummerGpuOutputFile(OPT_output_file) != 0)
   {
      exit(1);
   }

   Reference ref;
   if ((err = createReference(OPT_reffilename, &ref)))
   {
	  printStringForError(err);
	  exit(err);
   }
   
   QuerySet queries;
   if ((err = createQuerySet(OPT_qryfilename, &queries)))
   {
	  printStringForError(err);
	  exit(err);
   }

   MatchContext ctx;
   if ((err = createMatchContext(&ref, 
								&queries, 
								0, 
								OPT_on_cpu, 
								OPT_match_length, 
								OPT_stats_file,
								OPT_reverse,
                                OPT_forwardreverse,
                                OPT_forwardcoordinates,
                                OPT_showQueryLength,
								OPT_dotfilename,
                                OPT_texfilename,
								&ctx)))
   {
	  printStringForError(err);
	  exit(err);
   }   

   if ((err = matchQueries(&ctx)))
   {
	  printStringForError(err);
	  exit(err);
   }   
   closeMummerGpuOutputFile();
   MummerGpuOutputSummary output_summary;
   getMummerGpuOutputSummary(&output_summary);
   MummerGpuReference reference;
   populateReference(&reference, &ref_digest, &query_digest, &output_summary);
   int reference_status = 0;
   if (OPT_save_reference)
   {
      reference_status = saveReference(OPT_save_reference, &reference);
   }
   if (reference_status == 0 && OPT_verify_reference)
   {
      reference_status = verifyReference(OPT_verify_reference, &reference);
   }
   
   if ((err = destroyMatchContext(&ctx)))
   {
	  printStringForError(err);
	  exit(err);
   }   
   if (reference_status != 0)
   {
      if (OPT_verify_reference)
      {
         rodinia_print_fail("MUMmerGPU reference verification");
      }
      exit(1);
   }
}
