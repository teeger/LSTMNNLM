#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "nnlm/maxent_nnlm.h"

// <iostream>
using std::cout;
using std::cerr;
using std::endl;
// <fstream>
using std::ifstream;
// <string>
using std::string;
// <vector>
using std::vector;
// <boost/program_options.hpp>
namespace po = boost::program_options;

#define PROGRAM_NAME "Maximum Entropy (Neural Network) Langauge Model"
#define VERSION "0.0.4"

int main(int argc, char **argv) {

  string config_file;

  int debug;
  bool unk;
  bool nce_ppl;

  vector<string> trainfiles;
  string validationfile;
  string testfile;
  string vocfile;
  string outbase;
  string inmod;

  bool shuffle_datafiles;
  bool shuffle_sentences;

  float initalpha;
  int batch;
  float minimprovement;

  float beta;
  bool adagrad;

  bool nce;
  int num_negative_samples;

  bool globalbias;

  int ngram_order;
  int hash_table_size;
  int hash_mode;

  // Declare a group of options that will be 
  // allowed only on command line
  po::options_description generic("Generic options");
  generic.add_options()
      ("version,v", "print version information")
      ("help,h", "print help message")
      ("config,c", po::value<string>(&config_file)->default_value(""),
       "name of a file of a configuration (can be overwritten by command line options)")
      ;

  // Declare a group of options that will be 
  // allowed both on command line and in
  // config file
  po::options_description config("Configuration");
  config.add_options()
      ("debug", po::value<int>(&debug)->default_value(0),
       "debug level")
      ("unk", po::value<bool>(&unk)->default_value(false)->implicit_value(true),
       "<unk> as valid token")
      ("nce-ppl", po::value<bool>(&nce_ppl)->default_value(false)->implicit_value(true),
       "only calculate unnormalized perplexity?")
      ("trainfiles", po::value<vector<string>>(&trainfiles)->multitoken(),
       "training data file(s) (OOVs replaced by <unk>)")
      ("validationfile", po::value<string>(&validationfile)->default_value(""),
       "validation data")
      ("testfile", po::value<string>(&testfile)->default_value(""),
       "test data")
      ("vocfile", po::value<string>(&vocfile)->default_value(""),
       "vocabulary file (including </s> and <unk>)")
      ("outbase", po::value<string>(&outbase)->default_value(""),
       "basename for outbase.model")
      ("inmodel", po::value<string>(&inmod)->default_value(""),
       "name of the model to use in testing mode")
      ("shuffle-datafiles", po::value<bool>(&shuffle_datafiles)->default_value(false)->implicit_value(true),
       "shuffle training data files at the beginning of each Epoch")
      ("shuffle-sentences", po::value<bool>(&shuffle_sentences)->default_value(false)->implicit_value(true),
       "shuffle sentences within each training data file.")
      ("init-alpha", po::value<float>(&initalpha)->default_value(0.1f),
       "initial learning rate")
      ("batch-size", po::value<int>(&batch)->default_value(1),
       "update parameters after this many words")
      ("min-improvement", po::value<float>(&minimprovement)->default_value(1.0),
       "minimum improvement scale for the log-likelihood")
      ("beta", po::value<float>(&beta)->default_value(0.0f),
      "l2 regularization parameter (per batch*token); it should be scaled according to the number of tokens in the training data")
      ("adagrad", po::value<bool>(&adagrad)->default_value(false)->implicit_value(true),
       "use AdaGrad (required for NCE in current version)")
      ("nce", po::value<bool>(&nce)->default_value(false)->implicit_value(true),
       "use noise contrastive estimation")
      ("nce-samples", po::value<int>(&num_negative_samples)->default_value(0),
       "number of negative samples per history in gradient computation (valid when nce is set")
      ("globalbias", po::value<bool>(&globalbias)->default_value(false)->implicit_value(true),
       "use globalbias for the last layer (required fo NCE, recommended when --bias is set)")
      ("ngram", po::value<int>(&ngram_order)->default_value(1),
       "ngram order")
      ("hash", po::value<int>(&hash_table_size)->default_value(0),
       "size of the hash table")
      ("hash-mode", po::value<int>(&hash_mode)->default_value(0),
       "hash mode: 0 - faster but less accurate; 1 - slower but more accurate")
      ;

  po::options_description cmdline_options;
  cmdline_options.add(generic).add(config);

  po::options_description config_file_options;
  config_file_options.add(config);

  po::variables_map vm;
  try {
    store(parse_command_line(argc, argv, cmdline_options), vm);
  } catch (po::error& e) {
    cerr << "ERROR: " << e.what() << endl;
    cerr << cmdline_options << endl;
    return EXIT_FAILURE;
  }
  notify(vm);

  if (vm.count("help")) {
    cout << PROGRAM_NAME << endl;
    cout << cmdline_options << endl;
    return EXIT_SUCCESS;
  }
  if (vm.count("version")) {
    cout << PROGRAM_NAME << endl;
    cout << "version: " << VERSION << endl;
    return EXIT_SUCCESS;
  }

  if (config_file != "") {
    ifstream ifs(config_file.c_str());
    if (!ifs) {
      cerr << "can not open config file: " << config_file << endl;
      return 0;
    } else {
      try {
        store(parse_config_file(ifs, config_file_options), vm);
      } catch (po::error& e) {
        cerr << "ERROR: " << e.what() << endl;
        cerr << config_file_options << endl;
        return EXIT_FAILURE;
      }
    }

    // command-line options overwrite config file options
    store(parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
  }

  nnlm::MaxEntNeuralNetLM maxent_nnlm;
  maxent_nnlm.set_debug(debug);
  maxent_nnlm.set_unk(unk);

  if (inmod != "") {
    if (testfile == "") {
      cerr << "Confused: reading a model, but no test data is specified" << endl;
      return EXIT_FAILURE;
    }
    maxent_nnlm.ReadLM(inmod);
    maxent_nnlm.EvalLM(testfile, nce_ppl);
    return EXIT_SUCCESS;
  }

  if (vocfile == "") {
    cerr << "--vocfile not specified!" << endl;
    return EXIT_FAILURE;
  }
  if (outbase == "") {
    cerr << "--outbase not specified!" << endl;
    return EXIT_FAILURE;
  }
  if (testfile != "") {
    cerr << "--testfile specified in train mode!" << endl;
    return EXIT_FAILURE;
  }

  maxent_nnlm.set_vocab_filename(vocfile);
  maxent_nnlm.set_train_filenames(trainfiles);
  maxent_nnlm.set_shuffle_datafiles(shuffle_datafiles);
  maxent_nnlm.set_shuffle_sentences(shuffle_sentences);
  maxent_nnlm.set_algopts(initalpha, batch, minimprovement);
  maxent_nnlm.set_l2_regularization_param(beta);
  maxent_nnlm.set_adagrad(adagrad);

  maxent_nnlm.set_nce(nce);
  maxent_nnlm.set_num_negative_samples(num_negative_samples);

  // must reset activations at the beginning of every sentence!
  maxent_nnlm.set_independent(true);
  maxent_nnlm.set_globalbias(globalbias);
  // no need to set bias and errorinput_cutoff as they are not used.

  maxent_nnlm.set_ngram_order(ngram_order);
  maxent_nnlm.set_hash_table_size(hash_table_size);
  maxent_nnlm.set_hash_mode(hash_mode);

  maxent_nnlm.TrainLM(validationfile, outbase, nce_ppl);
}
