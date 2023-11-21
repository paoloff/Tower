from classes import *
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

assets = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "NFLX", "DIS", "META"]
risk_free_rate = .00
interval = "1m"
period = "5d"
initial_cash = 10000
steps_per_episode = 180
rois = []
cc = 1
n_episodes = 100
print_freq = steps_per_episode/10

while cc < n_episodes+1:
    simulator = MarketSimulator(initial_cash, assets, np.zeros(len(assets)), risk_free_rate, interval, period, steps_per_episode)
    robot = RfLearner(assets, batch_size=10)
    nav = initial_cash
    cash = initial_cash
    allocations = np.zeros(len(assets))
    prices = np.zeros(len(assets))
    nav_before = initial_cash
    simulator.reset(initial_cash, assets, np.zeros(len(assets)))
    start_time = simulator.time_step
    print("\n=================== New episode ===================")
    _, prices, allocations, cash, _ = simulator.step(np.zeros(len(assets)))
    print(allocations, cash)
    allocations = torch.tensor(allocations)
    open_prices = prices
    prices = torch.tensor(prices)
    cash = torch.tensor([cash])
    state = torch.cat((torch.stack((prices, allocations), 1).flatten(), cash)).to(dtype=torch.float32)

    for j in range(steps_per_episode):
        action = int(robot.epsilon_greedy(state))
        action = -np.mod(np.array((list(np.base_repr(action,3))), dtype = np.float32),3) + 1
        if len(action) < simulator.n_assets: action = np.pad(action, (simulator.n_assets - len(action), 0))
        reward, prices, allocations, cash, nav = simulator.step(action)
        if j % print_freq ==0: 
            print("\nStep number", j)
            print("Allocations", allocations, " || Cash", np.round(cash,1))
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
    print("\nAll ROI's", np.round(np.array(rois),1), "\nNext episode number =", cc+1, "\n")
    cc+=1

np.savetxt("rois.txt", rois, fmt = '%.2f')