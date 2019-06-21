import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run MixHop/N-GCN.")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--features-path",
                        nargs = "?",
                        default = "./input/features.csv",
	                help = "Features json.")

    parser.add_argument("--target-path",
                        nargs = "?",
                        default = "./input/target.csv",
	                help = "Target classes csv.")

    parser.add_argument("--clustering-method",
                        nargs = "?",
                        default = "metis",
	                help = "Target classes csv.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
	                help = "Number of training epochs. Default is 5.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for train-test split. Default is 42.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.1,
	                help = "Dropout parameter. Default is 0.1.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--threshold",
                        type = float,
                        default = 0.5,
	                help = "Learning rate. Default is 0.5.")

    parser.add_argument("--test-ratio",
                        type = float,
                        default = 0.1,
	                help = "Weight cut-off. Default is 0.1.")

    parser.add_argument("--cluster-number",
                        type = int,
                        default = 50,
                        help = "Architecture neuron allocation budget. Default is 60.")

    parser.set_defaults(layers = [512, 256, 128])
    
    return parser.parse_args()

