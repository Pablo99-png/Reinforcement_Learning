import json
import matplotlib.pyplot as plt

def read_json_file(file_path):
    elements = []
    with open(file_path, 'r') as file:
        for line in file:
            elements.append(json.loads(line))
    return elements

def correct_json_file(file_path, switch_name=False):
    with open(file_path, 'r') as file:
        with open(file_path.replace('.json', '_mod.json'), 'w') as file2:
            n_s = file.read().replace('}{', '}\n{')
            if switch_name:
                n_s = n_s.replace('meanReward', 'best_reward')
            file2.write(n_s)

def plot_data(data):
    rewards = [item['best_reward'] for item in data]
    times = [item['time'] for item in data]
    
    plt.plot(times, rewards, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title('Reward vs Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # ES
    file_path = './DIP_ES/REWARDS_ES.json'  # Replace with your JSON file path
    correct_json_file(file_path)
    data = read_json_file(file_path.replace('.json', '_mod.json'))
    plot_data(data)

    # PPO
    file_path = './DIP_PPO/REWARDS_PPO.json'  # Replace with your JSON file path
    correct_json_file(file_path, switch_name=True)
    data = read_json_file(file_path.replace('.json', '_mod.json'))
    plot_data(data)