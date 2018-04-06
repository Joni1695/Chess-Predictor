import numpy as np 
import pandas as pd
from pandas import Series, DataFrame

import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from sklearn import metrics

import chess
import chess.pgn

# Preprocessing Data Functions

def turnBoolInt(x):
	if(x == True):
		return 1
	return 0

def ecoToId(x):
	return uniqueECO[uniqueECO['eco']== x].index.values.astype(int)[0]

def winnerToId(x):
	return uniquetarget[uniquetarget['winner'] == x].index.values.astype(int)[0]

def getMat(x):
	return x[0]

def getMob(x):
	return x[1]

def getCC(x):
	return x[2]

def getPS(x):
	return x[3]

def boardEvaluation(x):
	# Load the game into the board, this is for easier evaluation
	board = chess.Board()
	xSplit = x.split()

	for move in xSplit:
		board.push_san(move)

	# Apply evaluation board
	# Material Count
	bRook = 0
	wRook = 0
	bQueen = 0
	wQueen = 0
	bKnight = 0
	wKnight = 0
	bOfficer = 0
	wOfficer = 0
	bPawn = 0
	wPawn = 0
	bIsolatedPawn = 0
	wIsolatedPawn = 0
	bDoublePawn = 0
	wDoublePawn = 0
	bBlockedPawn = 0
	wBlockedPawn = 0

	bPiecesUnicodeSymbols = [chess.Piece(4,chess.BLACK).unicode_symbol(),chess.Piece(3,chess.BLACK).unicode_symbol(),chess.Piece(2,chess.BLACK).unicode_symbol(),chess.Piece(5,chess.BLACK).unicode_symbol(),chess.Piece(6,chess.BLACK).unicode_symbol()]
	wPiecesUnicodeSymbols = [chess.Piece(4,chess.WHITE).unicode_symbol(),chess.Piece(3,chess.WHITE).unicode_symbol(),chess.Piece(2,chess.WHITE).unicode_symbol(),chess.Piece(5,chess.WHITE).unicode_symbol(),chess.Piece(6,chess.WHITE).unicode_symbol()]


	for i in chess.SQUARES:
		if board.piece_at(i) is not None:
			if board.piece_at(i).unicode_symbol() == chess.Piece(4,chess.WHITE).unicode_symbol():
				wRook = wRook + 1
			elif board.piece_at(i).unicode_symbol() == chess.Piece(4,chess.BLACK).unicode_symbol():
				bRook = bRook + 1
			elif board.piece_at(i).unicode_symbol() == chess.Piece(3,chess.WHITE).unicode_symbol():
				wOfficer = wOfficer + 1	
			elif board.piece_at(i).unicode_symbol() == chess.Piece(3,chess.BLACK).unicode_symbol():
				bOfficer = bOfficer + 1		
			elif board.piece_at(i).unicode_symbol() == chess.Piece(5,chess.WHITE).unicode_symbol():	
				wQueen = wQueen + 1	
			elif board.piece_at(i).unicode_symbol() == chess.Piece(5,chess.BLACK).unicode_symbol():
				bQueen = bQueen + 1		
			elif board.piece_at(i).unicode_symbol() == chess.Piece(2,chess.WHITE).unicode_symbol():
				wKnight = wKnight + 1		
			elif board.piece_at(i).unicode_symbol() == chess.Piece(2,chess.BLACK).unicode_symbol():
				bKnight = bKnight + 1		
			elif board.piece_at(i).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
				wPawn = wPawn + 1
				# Check for blocked pawns
				if i < 56 :
					if board.piece_at(i+8) is not None:
						if board.piece_at(i+8).unicode_symbol() in bPiecesUnicodeSymbols :
							wBlockedPawn = wBlockedPawn + 1
				
				# Check for doubled pawns on a single file
				PawnColumn = i % 8
				nrPawn = 0
				for j in range(0,8):
					if board.piece_at(PawnColumn + (j * 8)) == 	chess.Piece(1,chess.WHITE).unicode_symbol():
						nrPawn = nrPawn + 1
				if nrPawn > 1:
					wDoublePawn = wDoublePawn + 1

				# Check for isolated pawns
				nrPawn = 0
				PawnRow = int(i // 8)
				if PawnColumn != 0:
					if board.piece_at(i-1) is not None:
						if board.piece_at(i-1).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7:
					if board.piece_at(i+1) is not None:
						if board.piece_at(i+1).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnRow != 0:
					if board.piece_at(i-8) is not None:
						if board.piece_at(i-8).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7:
					if board.piece_at(i+8) is not None:
						if board.piece_at(i+8).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 0 and PawnRow != 0:
					if board.piece_at(i-9) is not None:
						if board.piece_at(i-9).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7 and PawnRow != 7:
					if board.piece_at(i+9) is not None:
						if board.piece_at(i+9).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 0 and PawnRow != 7:
					if board.piece_at(i-7) is not None:
						if board.piece_at(i-7).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7 and PawnRow != 0:
					if board.piece_at(i+7) is not None:
						if board.piece_at(i+7).unicode_symbol() == chess.Piece(1,chess.WHITE).unicode_symbol():
							nrPawn = nrPawn + 1
				if nrPawn == 0:
					wIsolatedPawn = wIsolatedPawn + 1		
			elif board.piece_at(i).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
				bPawn = bPawn + 1
				if i > 7 :
					if board.piece_at(i-8) is not None:
						if board.piece_at(i-8).unicode_symbol() in wPiecesUnicodeSymbols :
							bBlockedPawn = bBlockedPawn + 1

				PawnColumn = i % 8
				nrPawn = 0
				for j in range(0,8):
					if board.piece_at(PawnColumn + (j * 8)) == 	chess.Piece(1,chess.BLACK).unicode_symbol():
						nrPawn = nrPawn + 1
				if nrPawn > 1:
					bDoublePawn = bDoublePawn + 1

				# Check for isolated pawns
				nrPawn = 0
				PawnRow = int(i // 8)
				if PawnColumn != 0:
					if board.piece_at(i-1) is not None:
						if board.piece_at(i-1).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7:
					if board.piece_at(i+1) is not None:
						if board.piece_at(i+1).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnRow != 0:
					if board.piece_at(i-8) is not None:
						if board.piece_at(i-8).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7:
					if board.piece_at(i+8) is not None:
						if board.piece_at(i+8).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 0 and PawnRow != 0:
					if board.piece_at(i-9) is not None:
						if board.piece_at(i-9).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7 and PawnRow != 7:
					if board.piece_at(i+9) is not None:
						if board.piece_at(i+9).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 0 and PawnRow != 7:
					if board.piece_at(i-7) is not None:
						if board.piece_at(i-7).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if PawnColumn != 7 and PawnRow != 0:
					if board.piece_at(i+7) is not None:
						if board.piece_at(i+7).unicode_symbol() == chess.Piece(1,chess.BLACK).unicode_symbol():
							nrPawn = nrPawn + 1
				if nrPawn == 0:
					bIsolatedPawn = bIsolatedPawn + 1

	MaterialCount = (wQueen-bQueen) + (wRook-bRook) + ((wOfficer-bOfficer)+(wKnight-bKnight)) + (wPawn-bPawn)
	# Mobility
	board.turn = chess.WHITE
	wMobility = len(board.legal_moves)
	board.turn = chess.BLACK
	bMobility = len(board.legal_moves)

	MobilityCount = (wMobility - bMobility)
	# Pawn Structure
	PawnStructure = (wDoublePawn - bDoublePawn) + (wIsolatedPawn - bIsolatedPawn) + (wBlockedPawn - bBlockedPawn)
	# Centre Control
	CentralControl = (len(board.attackers(chess.WHITE,35)) - len(board.attackers(chess.BLACK,35))) + (len(board.attackers(chess.WHITE,36)) - len(board.attackers(chess.BLACK,36))) + (len(board.attackers(chess.WHITE,27)) - len(board.attackers(chess.BLACK,27))) + (len(board.attackers(chess.WHITE,28)) - len(board.attackers(chess.BLACK,28))) + (len(board.attackers(chess.WHITE,19)) - len(board.attackers(chess.BLACK,19))) + (len(board.attackers(chess.WHITE,20)) - len(board.attackers(chess.BLACK,20))) + (len(board.attackers(chess.WHITE,29)) - len(board.attackers(chess.BLACK,29))) + (len(board.attackers(chess.WHITE,37)) - len(board.attackers(chess.BLACK,37))) + (len(board.attackers(chess.WHITE,43)) - len(board.attackers(chess.BLACK,43))) + (len(board.attackers(chess.WHITE,44)) - len(board.attackers(chess.BLACK,44))) + (len(board.attackers(chess.WHITE,34)) - len(board.attackers(chess.BLACK,34))) + (len(board.attackers(chess.WHITE,26)) - len(board.attackers(chess.BLACK,26)))
	# King Safety
	return [MaterialCount,MobilityCount,CentralControl,PawnStructure]

# Make NumPy print data fully
np.set_printoptions(threshold=np.nan)

# Read the dataset
total_data = pd.read_csv('games.csv')

# Preprocessing
# Transform start and finish variables into a single variable, defined as finish - start

start = total_data['created_at']
finish = total_data['last_move_at']

finalTime=(finish-start)/1000

# Transform white_rating and black_rating variables into a single variable, defined as wRating - bRating

wRating = total_data['white_rating']
bRating = total_data['black_rating']

finalRating = wRating -bRating

# Drop the columns that are not going to be included in the model

total_data = total_data.drop(['id','created_at','victory_status','last_move_at','increment_code','white_id','black_id','opening_name','opening_ply','white_rating','black_rating'],axis = 1)

# Add the new columns we created above

total_data = pd.concat([total_data,finalTime,finalRating], axis = 1);
total_data.columns=['rated','nr_turns','winner','moves','opening_eco','final_time','final_rating']

# Fix issue in dataset, certain games have a 0 finalTime, this is an error caused by bad data
# We fix it by finding the mean value of time / turn, and applying that to the bad data.

select_data= total_data.loc[total_data['final_time'] != 0]

p = total_data.loc[total_data['final_time'] == 0]
for i,row in p.iterrows():
	p.set_value(i, 'final_time',row['nr_turns']*25) 


# Remove the target from the dataset

target_values = total_data['winner']
total_data = total_data.drop('winner',axis = 1)

# Turn rated into 1 or 0, again to be able to apply models like logistic regression
rated_data = total_data['rated'].apply(turnBoolInt)
total_data = total_data.drop('rated', axis = 1)
total_data = pd.concat([total_data,rated_data],axis =1)

# Create dummy variables for our opening_eco, remove one of the possible values to avoid multilinnearity

uniqueECO = total_data['opening_eco'].unique()
uniqueECO = pd.DataFrame({'eco':uniqueECO.tolist()})
ecoId = total_data['opening_eco'].apply(ecoToId)

total_data = total_data.drop('opening_eco', axis = 1)
total_data = pd.concat([total_data,ecoId],axis =1)

uniquetarget = target_values.unique()
uniquetarget = pd.DataFrame({'winner':uniquetarget.tolist()})
target_values_id = target_values.apply(winnerToId)

total_data = total_data.drop('final_time',axis = 1)
total_data = total_data.drop('nr_turns',axis = 1)

# Preprocess the game variable

move_data = total_data['moves'].apply(boardEvaluation)
material_data = move_data.apply(getMat)
mobility_data = move_data.apply(getMob)
central_data = move_data.apply(getCC)
pawn_data = move_data.apply(getPS)
total_data = total_data.drop('moves', axis = 1)
total_data = pd.concat([total_data,material_data,mobility_data,central_data,pawn_data],axis =1)
total_data.columns=['Final Rating','Rated Game','Opening Identifier','Material Count','Mobility Count','CentralControl','PawnStructure']

# Train the model and test the testing set

for i in range(1,100):
	X_train, X_test, Y_train, Y_test = train_test_split(total_data, target_values_id, test_size = 0.5)
	regr = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(7, 3), random_state=1)
	regr.fit(X_train, Y_train)
	print regr.score(X_test, Y_test, sample_weight=None)
