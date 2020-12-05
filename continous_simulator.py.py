import random
import math
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from time import sleep
import matplotlib.pyplot as plt

def calculate_price(shares, win_share, lose_share, b, buy_sell, current_holding):
	old_cost = b * math.log(math.exp(win_share/b) + math.exp(lose_share/b))
	prev_y , prev_n = win_share, lose_share
	if buy_sell == 'buy':
		if current_holding < 0: 
			if shares >= abs(current_holding):
				lose_share -= abs(current_holding)
				win_share += shares - abs(current_holding)
			else:
				lose_share -= shares
		else:
			win_share += shares			
	else:
		if current_holding > 0:	
			if shares >= current_holding:
				win_share -= current_holding
				lose_share += shares - current_holding
			else:
				win_share -= shares
		else:
			lose_share += shares
	new_win = math.exp(win_share/b) / (math.exp(win_share/b) + math.exp(lose_share/b))
	new_cost = b * math.log(math.exp(win_share/b) + math.exp(lose_share/b))
	price_to_pay = new_cost - old_cost
	return (price_to_pay, new_win, win_share, lose_share, prev_y, prev_n)

def add_noise(win_prob, player_type):
	if player_type == 0:
		prob = random.gauss(win_prob, 0.1)
		if prob <= win_prob+0.1 or prob >= win_prob-0.1:
			return prob
		else:
			return add_noise(win_prob, player_type)
	elif player_type == 1:
		prob = random.gauss(win_prob, 0.2)
		if prob <= win_prob+0.25 or prob >= win_prob-0.25:
			return prob
		else:
			return add_noise(win_prob, player_type)	
	return None
			
def prob_to_shares(win_prob, current_win, b):
	shares = 0
	action = ''
	if win_prob >= current_win:
		shares = abs((current_win - win_prob)/((1.01 - win_prob)*0.3)) * (b/5) + random.randint(1,5)
		action = 'buy'
	else:
		shares = abs((current_win - win_prob)/((0.01 + win_prob)*0.3)) * (b/5) + random.randint(1,5)
		action = 'sell'
	return round(shares), action
	
def market_resolve(final, df):
	mm_profit = 0
	for i in df.index:
		current_holding = df.iloc[i, 2]
		current_balance = float(df.iloc[i, 3])
		df.at[i, 'Shares'] = 0
		if final == 'win':
			if current_holding > 0:
				current_balance += current_holding
		else:
			if current_holding < 0:
				current_balance += abs(current_holding)
		df.at[i, 'Balance'] = str(current_balance)
		df.at[i, 'Profit'] = current_balance - float(init_balance)
		mm_profit += current_balance 
	mm_profit = float(init_balance)*df.shape[0] - mm_profit
	return mm_profit, df

team = input('Which team\'s winning probability the market maker wants to predict?: ')
total_win = float(input('Enter the number of won matches: '))
total = float(input('Enter the number of total matches: '))
win_ratio = total_win/total
position = int(input('Enter the current position of the team on points table: '))
max_bound = float(input('Enter the worst case loss market maker can afford: '))

print('\nIPL Prediction Market Simulation')
print('********************************\n')
print('******************************************************************************************')
print('*        Question: Will',team,'win IPL 2020 after observing league stage point table?        *')
print('******************************************************************************************\n\n')

current_win = 0.5									#current win probability
win_share = 0										#outstanding shares of winning
lose_share = 0										#outstanding shares of losing
b = abs(float(max_bound/math.log(2)))				#constant to bound the loss of market maker
playerType = ['Expert', 'Medium', 'Novice']
df = pd.read_csv('user_info.csv')
data = pd.read_csv("TrainData.csv")
X = np.array(data.iloc[:,:-1])
y = np.array(data.iloc[:,-1]) 
test = [[win_ratio, position]]
dtc = tree.DecisionTreeClassifier()
dtc.fit(X,y)
init_balance = str(2 * b)
print('Win probability: ', current_win, '\n')	
for it in range(50):
	print(it,'\n')
	playerTypeCount = [10, 10, 10]
	users = [i for i in range(1,sum(playerTypeCount)+1)]
	for player in range(sum(playerTypeCount)):
		if it == 0:
			chosenPlayer = random.randint(0,2)
			while playerTypeCount[chosenPlayer] <= 0:
				chosenPlayer = random.randint(0,2)		
			playerTypeCount[chosenPlayer]-=1
			user_id = df.shape[0] + 1
			pType = playerType[chosenPlayer]
			df = df.append(pd.DataFrame({"Player_id" : [user_id,], "Player_Type" : [pType,], "Shares" : [0,], "Balance" : [init_balance,], "Profit" : [0,]}), sort=False, ignore_index=True)
		
		elif it != 0:
			random.shuffle(users)
			user_id = users[player]
			pType = df.iloc[user_id-1, 1]
			
		current_holding = df.iloc[user_id-1, 2]
		current_balance = float(df.iloc[user_id-1, 3])
		
		chances = None
		win_prob = current_win
		if chosenPlayer == 0 or chosenPlayer == 1:
			if chosenPlayer == 0:
				chances = dtc.predict_proba(test)
			else:
				chances = dtc.predict_proba(test)
			
			win_prob = np.amax(chances)
			win_lose = np.argmax(chances)
			
			if win_lose == 2:
				win_prob = 0.8*chances[0][1] + 0.2*chances[0][2]
			elif win_lose == 0:
				win_prob = 1 - win_prob
			
			win_prob = add_noise(win_prob, chosenPlayer)
			win_prob = win_prob if win_prob<=1 else 2-win_prob
			win_prob = win_prob if win_prob>=0 else abs(win_prob)
		else:
			win_prob = np.random.rand()
		
		print('Expected Player probability: ',win_prob,'\n')
		
		shares, action = prob_to_shares(win_prob, current_win, b)
		
		price_to_pay, new_win, win_share, lose_share, prev_y, prev_n = calculate_price(shares, win_share, lose_share, b, action, current_holding)
		
		if current_balance >= price_to_pay:
			current_win = new_win
			if action == 'buy':
				current_holding += shares
				current_balance -= price_to_pay
			else:
				current_holding -= shares
				current_balance -= price_to_pay
		else:
			win_share = prev_y
			lose_share = prev_n
			
		print('UserId:', user_id, pType, 'player bought/sold', shares, 'shares\n')
		print('Win_share :', win_share, 'Lose_share :', lose_share, '\n')
		
		df.at[user_id-1, 'Shares'] = current_holding
		df.at[user_id-1, 'Balance'] = str(current_balance)

		df.to_csv('user_info.csv', index=False)
		print(df)
		print('\n')     
	print('Win probability: ', current_win, '\n')

print('Prediction Time Over')
print('********************\n')
final = input('End result? Enter win or lose: ')
mm_profit, df = market_resolve(final, df)
df.to_csv('user_info_1.csv', index=False)
print('\nMarket Resolved\n')
print('Market Maker\'s Profit: ', mm_profit, '\n')
print(df, '\n\n')

X = np.array(df.iloc[:,-1])
y = np.array(df.iloc[:,1])
expert = X[y=='Expert']
novice = X[y=='Novice']
medium = X[y=='Medium']
plt.boxplot([expert,medium,novice])
plt.xticks([1, 2, 3], ['Expert', 'Medium', 'Novice'])
plt.savefig('Continuous.png')

