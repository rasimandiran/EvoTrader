# Agent for DQN

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Trader(object):
    def __init__(self, brain, capital, row, gen):
        self.brain = brain
        self.brain.trader = self
        self.gen = gen
        self.row = row
        self.name = "G"+str(gen)+"_T"+str(row)
        self.reward = 0
        self.rewards = []
        self.state = {
            "action": 0,
            "reward": 0,
            "state": [],
            "next_state": [],
            "done": False,
            "position": "",
            "position_price": 0,
            "max_price_after_position": 0,
            "min_price_after_position": 0,
        }

        self.init_capital = capital
        self.capital = capital
        self.portfolio = [self.capital]
        self.quote_balance = self.capital
        self.asset_balance = 0

        self.wins = []
        self.wins_r = []
        self.cons_wins = 0
        self.max_cons_wins = 0

        self.losses = []
        self.losses_r = []
        self.cons_loss = 0
        self.max_cons_loss = 0

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}"

    def __zero_dev(self, success, failure):
        try:
            return success()
        except:
            return failure

    def reset(self):
        self.reward = 0
        self.rewards = []
        self.state = {
            "action": 0,
            "reward": 0,
            "state": [],
            "next_state": [],
            "done": False,
            "position": "",
            "position_price": 0,
            "max_price_after_position": 0,
            "min_price_after_position": 0,
        }

        self.capital = self.init_capital
        self.portfolio = [self.capital]
        self.quote_balance = self.capital
        self.asset_balance = 0

        self.wins = []
        self.wins_r = []
        self.cons_wins = 0
        self.max_cons_wins = 0

        self.losses = []
        self.losses_r = []
        self.cons_loss = 0
        self.max_cons_loss = 0
        return

    def save(self, episode):
        path = f"episodes/episode_{str(episode)}/gen_{self.gen}/trader_{self.name}/"
        os.makedirs(path, exist_ok=True)

        #brain
        brain = {
            "lr": self.brain.lr,
            "gamma": self.brain.gamma,
            "epsilon": self.brain.epsilon,
            "epsilon_decay": self.brain.epsilon_decay,
            "target_update_every": self.brain.target_update_every
        }
        self.brain.neural_network.save(path+"neural_network.h5")
        with open(path+"brain.json", "w", encoding="utf8") as f:
            f.write(json.dumps(brain, indent=4, ensure_ascii=False))

        #stats
        stats = {
            "Toplam işlem sayısı": self.__zero_dev(lambda: len(self.wins) + len(self.losses), 0),
            "Karlı işlem sayısı": self.__zero_dev(lambda: len(self.wins), 0),
            "Ortalama kar": self.__zero_dev(lambda: round(sum(self.wins)/len(self.wins),2), 0),
            "Ortalama kar (%)": self.__zero_dev(lambda: round(sum(self.wins_r)/len(self.wins_r), 2), 0),
            "Max kazanç": self.__zero_dev(lambda: round(max(self.wins), 2), 0),
            "Max kazanç (%)": self.__zero_dev(lambda: round(max(self.wins_r), 2), 0),
            "Üst üste kazanç": self.__zero_dev(lambda: self.max_cons_wins, 0),
            "Toplam kazanç": self.__zero_dev(lambda: round(sum(self.wins), 2), 0),
            "Zararlı işlem sayısı": self.__zero_dev(lambda: len(self.losses), 0),
            "Ortalama zarar": self.__zero_dev(lambda: round(sum(self.losses)/len(self.losses),2) , 0),
            "Ortalama zarar (%)": self.__zero_dev(lambda: round(sum(self.losses_r)/len(self.losses_r), 2), 0),
            "Max zarar": self.__zero_dev(lambda: round(min(self.losses), 2), 0),
            "Max zarar (%)": self.__zero_dev(lambda: round(min(self.losses_r), 2), 0),
            "Üst üste zarar": self.__zero_dev(lambda: self.max_cons_loss, 0),
            "Toplam zarar": self.__zero_dev(lambda: round(sum(self.losses), 2), 0),
            "Kasa en düşük": self.__zero_dev(lambda: round(min(self.portfolio), 2), 0),
            "Kasa en yüksek": self.__zero_dev(lambda: round(max(self.portfolio), 2), 0),
            "Kasa son": self.__zero_dev(lambda: round(self.capital, 2), 0),
            "Risk-getiri": self.__zero_dev(lambda: round(-1*((sum(self.wins_r)/len(self.wins_r))/(sum(self.losses_r)/len(self.losses_r))), 2), 0),
            "PnL": self.__zero_dev(lambda: round(self.capital-self.init_capital, 2), 0),
            "PnL (%)": self.__zero_dev(lambda: round(((self.capital-self.init_capital)/self.init_capital)*100, 2), 0),
            "Reward": self.__zero_dev(lambda: round(self.reward, 2), 0)
        }
        with open(path+"stats.json", "w", encoding="utf8") as f:
            f.write(json.dumps(stats, indent=4, ensure_ascii=False))

        #portfolio chart
        fig, ax = plt.subplots(1, 1, figsize=(50, 20))
        ax.set_title("Kasa")
        ax.axhline(self.init_capital, color="#000000", ls="-.", linewidth=0.5)
        ax.plot(self.portfolio, marker="o", markersize=12)
        fig.savefig(path+"portfolio.png")
        plt.close(fig)
        
        #reward chart
        fig, ax = plt.subplots(1, 1, figsize=(50, 20))
        ax.set_title("Reward")
        ax.plot(self.rewards, marker="o", markersize=12)
        fig.savefig(path+"rewards.png")
        plt.close(fig)

        return

    def update_state(self, low, high, close):
        if high > self.state["max_price_after_position"] and self.state["position"] == "sell":
            self.state["max_after_position"] = high
            penalty = ((self.state["max_price_after_position"]-self.state["position_price"]) / self.state["position_price"])*100
            self.state["reward"] = -penalty
        elif low < self.state["min_price_after_position"] and self.state["position"] == "buy":
            self.state["min_after_position"] = low
            penalty = ((self.state["position_price"]-self.state["min_price_after_position"]) / self.state["position_price"])*100
            self.state["reward"] = -penalty
        return

    def hold_state(self, prev_close, prev_state, close, current_state, done):
        inner_state_val = 0 if self.state["action"]==2 else 1
        inner_state = np.full([self.brain.state_space_size[0], self.brain.state_space_size[1], 1], inner_state_val)
        prev_state = np.dstack((prev_state, inner_state))
        current_state = np.dstack((current_state, inner_state))
        reward_or_penalty = prev_close-close if self.state["action"]==2 else close-prev_close
        self.state["reward"] += reward_or_penalty
        self.brain.experience(prev_state, 0, reward_or_penalty, current_state, done)
        return

    def change_state(self, price, amount, qty, state, action, next_state, done):        
        # buy -> sell
        if self.state["position"] == "buy" and action == 2:
            # new experience
            self.state["reward"] += ((price - self.state["position_price"])/self.state["position_price"])*100
            self.reward += self.state["reward"]
            self.rewards.append(self.reward)
            self.brain.experience(self.state["state"], self.state["action"], self.state["reward"], self.state["next_state"], self.state["done"])

            # wallets
            self.asset_balance += qty
            old_quote_balance = self.portfolio[-1]
            self.quote_balance += amount
            
            # trade stats
            pnl = self.quote_balance - old_quote_balance
            pnl_r = (pnl/old_quote_balance)*100
            
            self.capital += pnl 
            self.portfolio.append(self.capital)

            if pnl > 0:
                self.wins.append(pnl)
                self.wins_r.append(pnl_r)
                self.cons_loss = 0
                self.cons_wins += 1
                self.max_cons_wins = max(self.cons_wins, self.max_cons_wins)
            else:
                self.losses.append(pnl)
                self.losses_r.append(pnl_r)
                self.cons_wins = 0
                self.cons_loss += 1
                self.max_cons_loss = max(self.cons_loss, self.max_cons_loss)

            # new state
            inner_state = np.full([self.brain.state_space_size[0], self.brain.state_space_size[1], 1], 0)
            state = np.dstack((state, inner_state))
            next_state = np.dstack((next_state, inner_state))
            self.state = {
                "action": 2,
                "reward": 0,
                "state": state,
                "next_state": next_state,
                "done": done,
                "position": "sell",
                "position_price": price,
                "max_price_after_position": price,
                "min_price_after_position": price
            }

        # sell -> buy
        elif self.state["position"] == "sell" and action == 1:
            # new experience
            self.state["reward"] += ((self.state["position_price"]-price)/self.state["position_price"])*100
            self.reward += self.state["reward"]
            self.rewards.append(self.reward)
            self.brain.experience(self.state["state"], self.state["action"], self.state["reward"], self.state["next_state"], self.state["done"])

            # wallets
            self.asset_balance += qty
            self.quote_balance += amount

            # new state
            inner_state = np.full([self.brain.state_space_size[0], self.brain.state_space_size[1], 1], 1)
            state = np.dstack((state, inner_state))
            next_state = np.dstack((next_state, inner_state))
            self.state = {
                "action": 1,
                "reward": 0,
                "state": state,
                "next_state": next_state,
                "done": done,
                "position": "buy",
                "position_price": price,
                "max_price_after_position": price,
                "min_price_after_position": price
            }

        elif self.state["position"] == "" and action == 1:
            # wallets
            self.asset_balance += qty
            self.quote_balance += amount

            # new state
            inner_state = np.full([self.brain.state_space_size[0], self.brain.state_space_size[1], 1], 1)
            state = np.dstack((state, inner_state))
            next_state = np.dstack((next_state, inner_state))
            self.state = {
                "action": 1,
                "reward": 0,
                "state": state,
                "next_state": next_state,
                "done": done,
                "position": "buy",
                "position_price": price,
                "max_price_after_position": price,
                "min_price_after_position": price
            }
        
        return