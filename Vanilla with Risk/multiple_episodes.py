from classes_v2 import *
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


assets = ["AAPL", "DIS", "META","XOM","HSBC","PEP","JNJ","MCD"]
#assets = ["AAPL", "MSFT","TSLA"]
risk_free_rate = 10**(-6)
interval = "1m"
period = "5d"
initial_cash = 10000
steps_per_episode = 360
rois = []
multiple = 60
cc = 1
n_episodes = 100
print_freq = 20
min_threshold = 0.9
max_threshold = 2


while cc < n_episodes+1:
    simulator = MarketSimulator(initial_cash, assets, np.zeros(len(assets)), risk_free_rate, interval, period, multiple, steps_per_episode)
    robot = Learner(assets, batch_size=20)
    nav = initial_cash
    cash = initial_cash
    allocations = np.zeros(len(assets))
    prices = np.zeros(len(assets))
    nav_before = initial_cash
    simulator.reset(initial_cash, assets, np.zeros(len(assets)))
    start_time = simulator.time_step
    print("\n=================== New episode ===================")
    _, prices, allocations, cash, nav, risk = simulator.step(np.zeros(len(assets)))
    print("Starting with Allocations:", simulator.portfolio.allocations," Cash:", simulator.portfolio.cash, "and NAV:", nav)
    allocations = torch.tensor(allocations)
    open_prices = prices
    prices = torch.tensor(prices)
    cash = torch.tensor([cash])
    state = torch.cat((torch.stack((prices, allocations), 1).flatten(), cash)).to(dtype=torch.float32)


    for j in range(steps_per_episode):
        action = int(robot.epsilon_greedy(state))
        action = -np.mod(np.array((list(np.base_repr(action,3))), dtype = np.float32),3) + 1
        if len(action) < simulator.n_assets: action = np.pad(action, (simulator.n_assets - len(action), 0))
        reward, prices, allocations, cash, nav, risk = simulator.step(action)
        if nav <= min_threshold*initial_cash or nav > max_threshold*initial_cash: break
        if j % print_freq == 0: 
            print("\nStep number", j)
            print("Allocations", allocations, " || Cash", np.round(cash,1), " || Risk", np.round(risk,2), " || Reward", np.round(reward,2))
            print("NAV:", np.round(np.sum(allocations*prices)+cash,2))
        allocations = torch.tensor(allocations)
        prices = torch.tensor(prices)
        cash = torch.tensor([cash])
        next_state = torch.cat((torch.stack((prices, allocations), 1).flatten(), cash)).to(dtype=torch.float32)
        robot.memorize(state, action, reward, next_state)
        robot.experience_replay()
        state = next_state
    print("\n--------------------")
    print("--------------------")
    print("Start time:", start_time)
    print("Episode % ROI:", np.round(100*(nav/initial_cash-1),2), "\nEpisode NAV:", np.round(nav,2), "\nEpisode cash:", np.round(float(cash)))
    print("Episode allocations:", np.array(allocations))
    print("Episode open prices:", np.array(open_prices))
    print("Episode close prices:", np.array(prices))
    print("--------------------")
    print("--------------------\n")
    print("=================== End of episode ===================\n")
    rois.append(100*(nav/initial_cash-1))
    print("\nAll ROI's", np.array(rois),"\nAverage =", np.mean(np.array(rois)),"\nNext episode number =", cc+1, "\n")
    cc+=1


print("\n\nAverage ROI", np.mean(rois))
print("Standard deviation", np.std(rois))

np.savetxt("rois_risk_aversion_new.txt", rois, fmt = '%.2f')