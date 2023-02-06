from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--maxlen_train', type=int, default=60, help='Maximum number of tokens in the input sequence during training.')
parser.add_argument('--maxlen_test', type=int, default=60, help='Maximum number of tokens in the input sequence during evaluation.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training.')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam.')
parser.add_argument('--num_eps', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--task_type', type=int, default=0, help='0:Stance, 1:Sentiment')
parser.add_argument('--hidden_size', type=int, default=1024, help='hidden_size')
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='drop_out')
#parser.add_argument('--output_dir', type=str, default='my_model', help='Where to save the trained model, if relevant.')

args = parser.parse_args()