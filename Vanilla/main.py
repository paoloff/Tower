from classes import *
np.set_printoptions(precision=2)

assets = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "NFLX", "DIS", "META"]
initial_allocations = np.zeros(len(assets))
initial_cash = 10000
risk_free_rate = .001
initial_action = np.zeros(len(assets))
interval = "1m"
period = "5d"

n_episodes = 100
steps_per_episode = 50
navs = []

for k in range(n_episodes):
    simulator = MarketSimulator(initial_cash, assets, np.zeros(len(assets)), risk_free_rate, interval, period, steps_per_episode)
    robot = RfLearner(assets, batch_size=30)
    nav = initial_cash
    cash = initial_cash
    allocations = initial_allocations
    prices = np.zeros(len(assets))
    roi = 1
    nav_before = initial_cash
    simulator.reset(initial_cash, assets, np.zeros(len(assets)))
    print("Episode:", k)
    print("Start minute:", simulator.time_step)
    _, prices, allocations, cash, _ = simulator.step(initial_action)
    print(initial_action)
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
        
    navs.append(nav)
    
navs = np.array(navs)
rois = 100*(navs/initial_cash-1)

print("\n\n=====================\nRESULTS\n")
print("% ROI's", np.round(rois,1)) 
print("Average NAV", np.round(np.mean(navs),1))
print("Average % ROI", np.round(np.mean(100*(navs/initial_cash-1)),1))
print("Maximum % ROI", np.round(np.max(rois),1))
print("Minimum % ROI", np.round(np.min(rois),1))
print("Percentage of episodes with positive return", np.round(100*np.sum(rois>0)/n_episodes,1))
print("Average ROI for episodes with positive return", np.round(100*np.average(rois[rois>0])/n_episodes,1))
print("Average ROI for episodes with negative return", np.round(100*np.average(rois[rois<0])/n_episodes,1))
print("\n=====================\n\n")


