import cmd
import os

from core import data

class BirdSpeciesClassificationCLI(cmd.Cmd):
	prompt = 'bird-classification-species> '
	intro = 'Welcome to Bird Classification Species. Type "help" for available commands.'

	def __init__(self):
		super().__init__()

	def do_download(self, line):
		data.download_cub_200_2011()
		train_df, _, _ = data.get_dataframes_cub_200_2011()
		print(train_df)

	def do_quit(self, line):
		"""Exit the CLI."""
		return True

	def do_exit(self, line):
		"""Exit the CLI."""
		return True

	def postcmd(self, stop, line):
		# Add an empty line for better readability
		print()
		return stop

if __name__ == '__main__':
	BirdSpeciesClassificationCLI().cmdloop()
