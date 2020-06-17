import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from market import Market
from brain import Brain
from trader import Trader


EPISODES = 10
SURVIVOR_COUNT = 5
TRADER_COUNT_PER_GEN = 20 # GENERATION POPULATION
EVOLUTION_FREQ = 1

data = pq.read_pandas("data/BTC-USDT.parquet").to_pandas()
data.sort_values("open_time")
data = data.resample("1min").apply(dict(open="first", high="max", low="min", close="last", volume="sum"))
market = Market(data)

generation = 1
brains = [Brain(state_space_size=market.state_space_size, action_space_size=3) for _ in range(TRADER_COUNT_PER_GEN)]
traders = [Trader(brains[i], 10_000, i, generation) for i in range(TRADER_COUNT_PER_GEN)]

leaderboard = []

for episode in range(1, EPISODES+1):
    for step in tqdm(range(market.step_size), ncols=1000, position=0, leave=True):
        low, high, close, current_state, next_state = market.get_state(step)
        done = True if step == market.step_size-1 else False
        for trader in traders:
            trader.update_state(low, high, close)
            action = trader.brain.think(current_state)
            
            # buy
            if action == 1 and trader.quote_balance>100:
                asset_qty = market.buy(trader.quote_balance, close)
                trader.change_state(close, -trader.quote_balance, asset_qty, current_state, action, next_state, done)              

            # sell
            elif action == 2 and trader.asset_balance>0:
                amount = market.sell(trader.asset_balance, close)
                trader.change_state(close, amount, -trader.asset_balance, current_state, action, next_state, done)

            elif action == 0 and step>0:
                trader.hold_state(prev_close, prev_state, close, current_state, done)

            if done:
                leaderboard.append(trader)

        prev_close = close
        prev_state = current_state
    
    leaderboard.sort(key=lambda t: t.reward, reverse=True)
    print("==================================================EPISODE:"+str(episode)+"==================================================")
    for i, leader in enumerate(leaderboard):
        next_tour = "*" if (i+1)<=SURVIVOR_COUNT else ""
        print(f"{i+1}-){next_tour} Trader: {leader.name} Reward: {leader.reward} Pot: {leader.capital}")
    
    if episode % EVOLUTION_FREQ == 0:
        generation += 1
        traders = leaderboard[:SURVIVOR_COUNT]
        survivor_traders = leaderboard[:SURVIVOR_COUNT]
        print(survivor_traders)
        leaderboard = []

        for trader in survivor_traders:
            trader.save(episode)
            trader.reset()
            for i in range(3):
                mutated_brain = trader.brain.mutate()
                new_trader = Trader(mutated_brain, 10_000, str(trader.row)+"_"+str(i), generation)
                traders.append(new_trader)
