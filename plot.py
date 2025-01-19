import json
import matplotlib.pyplot as plt
import numpy as np

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

def plot_data(data, ppo):
    rewards = [item['best_reward'] for item in data]
    times = [item['time'] for item in data]

    if ppo:
      newTimes = [sum(times[:i]) for i in range(len(times))]
      newTimes = [newTimes[i*10] for i in range(len(times)//10) ] #da levare
      rewards = [rewards[i*10] for i in range(len(rewards)//10)] #da levare
      
    else:
      newTimes = times
    if ppo:
      plt.plot(newTimes, rewards, label='PPO',marker='o', markersize=3)
    else:
      plt.plot(times, rewards, label='ES',marker='o', markersize=3)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title('Reward vs Time')
    plt.grid(True)

if __name__ == "__main__":
    # ES
    file_path = '/content/drive/MyDrive/RL_FINAL/REWARDS_ES.json'  # Replace with your JSON file path
    correct_json_file(file_path)
    data = read_json_file(file_path.replace('.json', '_mod.json'))
    plot_data(data,False)

    # PPO
    file_path = '/content/drive/MyDrive/RL_FINAL/REWARDS_PPO.json'  # Replace with your JSON file path
    correct_json_file(file_path, switch_name=True)
    data = read_json_file(file_path.replace('.json', '_mod.json'))
    plot_data(data,True)



    plt.show()
