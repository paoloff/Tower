from classes import *
np.set_printoptions(precision=2)

assets = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "NFLX", "DIS", "META"]
risk_free_rate = .001
interval = "1m"
period = "5d"
initial_cash = 10000
steps_per_episode = 180
navs = []

simulator = MarketSimulator(initial_cash, assets, np.zeros(len(assets)), risk_free_rate, interval, period, steps_per_episode)
robot = RfLearner(assets, batch_size=10)
print("Length of ticker data", simulator.data_length)

nav = initial_cash
cash = initial_cash
allocations = np.zeros(len(assets))
prices = np.zeros(len(assets))
nav_before = initial_cash
simulator.reset(initial_cash, assets, np.zeros(len(assets)))
print("Start minute:", simulator.time_step)
_, prices, allocations, cash, _ = simulator.step(np.zeros(len(assets)))
allocations = torch.tensor(allocations)
prices = torch.tensor(prices)
cash = torch.tensor([cash])
state = torch.cat((torch.stack((prices, allocations), 1).flatten(), cash)).to(dtype=torch.float32)

for j in range(steps_per_episode):
    action = int(robot.epsilon_greedy(state))
    action = -np.mod(np.array((list(np.base_repr(action,3))), dtype = np.float32),3) + 1
    if len(action) < simulator.n_assets: action = np.pad(action, (simulator.n_assets - len(action), 0))
    reward, prices, allocations, cash, nav = simulator.step(action)
    allocations = torch.tensor(allocations)
    prices = torch.tensor(prices)
    cash = torch.tensor([cash])
    next_state = torch.cat((torch.stack((prices, allocations), 1).flatten(), cash)).to(dtype=torch.float32)
    robot.memorize(state, action, reward, next_state)
    robot.experience_replay()
    state = next_state
print("Episode NAV:", np.round(nav,2), "|| Episode % ROI:", np.round(100*(nav/initial_cash-1),2), "|| Episode cash:", np.round(float(cash)))
print("Episode allocations:", np.array(allocations))
print("Episode close prices:", np.array(prices))
print("--------------------")
print("--------------------")
    