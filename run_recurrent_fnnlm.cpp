#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "fnnlm/recurrent_fnnlm.h"

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

#define PROGRAM_NAME "Recurrent Factored Neural Network Langauge Model"
#define VERSION "0.0.1"

int main(int argc, char **argv) {

  string config_file;

  int debug;
  bool unk;
  bool nce_ppl;

  vector<string> trainfiles;
  string validationfile;
  string testfile;
  string word_vocfile;
  string factor_vocfile;
  string decompfile;
  string outbase;
  string inmod;
  string input_embedding;
  string output_embedding;

  bool shuffle_datafiles;
  bool shuffle_sentences;

  float initalpha;
  int batch;
  float minimprovement;

  float beta;
  bool adagrad;

  bool nce;
  int num_negative_samples;

  bool independent;
  bool globalbias;
  bool bias;
  float errorincutoff;

  bool use_factor_input;
  bool use_factor_hidden;
  float weight_factor_output;

  int bptt;
  int hidden;

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
      ("word-vocfile", po::value<string>(&word_vocfile)->default_value(""),
       "word vocabulary file (including </s> and <unk>)")
      ("factor-vocfile", po::value<string>(&factor_vocfile)->default_value(""),
      "factor vocabulary file (including </s> and <unk>)")
      ("decompfile", po::value<string>(&decompfile)->default_value(""),
       "word to factors decomposition file")
      ("outbase", po::value<string>(&outbase)->default_value(""),
       "basename for outbase.model")
      ("inmodel", po::value<string>(&inmod)->default_value(""),
       "name of the model to use in testing mode")
      ("input-embedding", po::value<string>(&input_embedding)->default_value(""),
       "extract input embedding from the model")
      ("output-embedding", po::value<string>(&output_embedding)->default_value(""),
       "extract output embedding from the model")
      ("shuffle-datafiles", po::value<bool>(&shuffle_datafiles)->default_value(false)->implicit_value(true),
       "shuffle training data files at the beginning of each Epoch")
      ("shuffle-sentences", po::value<bool>(&shuffle_sentences)->default_value(false)->implicit_value(true),
       "shuffle sentences within each training data file")
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
      ("independent", po::value<bool>(&independent)->default_value(false)->implicit_value(true),
       "reset activations after each sentence")
      ("globalbias", po::value<bool>(&globalbias)->default_value(false)->implicit_value(true),
       "use globalbias for the last layer (required fo NCE, recommended when --bias is set)")
      ("bias", po::value<bool>(&bias)->default_value(false)->implicit_value(true),
       "use bias for the last layer (recommend to set --globalbias as well)")
      ("errorin-cutoff", po::value<float>(&errorincutoff)->default_value(15),
       "error input cutoff")
      ("use-factor-input", po::value<bool>(&use_factor_input)->default_value(false)->implicit_value(true),
      "use factor input layer")
      ("use-factor-hidden", po::value<bool>(&use_factor_hidden)->default_value(false)->implicit_value(true),
      "use factor hidden layer")
      ("weight-factor-output", po::value<float>(&weight_factor_output)->default_value(0),
      "weight on factors in multi-task learning objective")
      ("bptt", po::value<int>(&bptt)->default_value(1),
       "bptt unfold level")
      ("hidden", po::value<int>(&hidden)->default_value(100),
       "hidden layer size")
      ("ngram", po::value<int>(&ngram_order)->default_value(0),
       "ngram order of the max-ent feature")
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

  fnnlm::RecurrentFNeuralNetLM recurrent_fnnlm;
  recurrent_fnnlm.set_debug(debug);
  recurrent_fnnlm.set_unk(unk);

  if (inmod != "") {
    if (testfile == "" && input_embedding == "" && output_embedding == "") {
      cerr << "Confused: reading a model, but no --testfile, --input-embedding, --output_embedding is specified" << endl;
      return EXIT_FAILURE;
    }
    recurrent_fnnlm.ReadLM(inmod);
    if (testfile != "") {
      recurrent_fnnlm.EvalLM(testfile, nce_ppl);
    }
    if (input_embedding != "") {
      recurrent_fnnlm.ExtractWordInputEmbedding(input_embedding);
    }
    if (output_embedding != "") {
      recurrent_fnnlm.ExtractWordOutputEmbedding(output_embedding);
    }
    return EXIT_SUCCESS;
  }

  if (word_vocfile == "") {
    cerr << "--word-vocfile not specified!" << endl;
    return EXIT_FAILURE;
  }
  if (factor_vocfile == "") {
    cerr << "--factor-vocfile not specified!" << endl;
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

  recurrent_fnnlm.set_word_vocab_filename(word_vocfile);
  recurrent_fnnlm.set_factor_vocab_filename(factor_vocfile);
  recurrent_fnnlm.set_decomp_filename(decompfile);
  recurrent_fnnlm.set_train_filenames(trainfiles);
  recurrent_fnnlm.set_shuffle_datafiles(shuffle_datafiles);
  recurrent_fnnlm.set_shuffle_sentences(shuffle_sentences);
  recurrent_fnnlm.set_algopts(initalpha, batch, minimprovement);
  recurrent_fnnlm.set_l2_regularization_param(beta);
  recurrent_fnnlm.set_l2_regularization_param(0);
  recurrent_fnnlm.set_adagrad(adagrad);

  recurrent_fnnlm.set_nce(nce);
  recurrent_fnnlm.set_num_negative_samples(num_negative_samples);

  recurrent_fnnlm.set_independent(independent);
  recurrent_fnnlm.set_globalbias(globalbias);
  recurrent_fnnlm.set_bias(bias);
  recurrent_fnnlm.set_errorinput_cutoff(errorincutoff);

  recurrent_fnnlm.set_use_factor_input(use_factor_input);
  recurrent_fnnlm.set_use_factor_hidden(use_factor_hidden);
  recurrent_fnnlm.set_weight_factor_output(weight_factor_output);

  recurrent_fnnlm.set_bptt_unfold_level(bptt);
  recurrent_fnnlm.set_nhiddens(hidden);

  recurrent_fnnlm.set_ngram_order(ngram_order);
  recurrent_fnnlm.set_hash_table_size(hash_table_size);
  recurrent_fnnlm.set_hash_mode(hash_mode);

  recurrent_fnnlm.TrainLM(validationfile, outbase, nce_ppl);
}
