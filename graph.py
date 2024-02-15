import matplotlib.pyplot as plt


def plot_graph(filename):
    file_sizes = []
    cpu_times = []
    gpu_times = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            filename = parts[0].split(': ')[1]
            size = int(parts[1].split(': ')[1].split(' ')[0])
            cpu_time = float(parts[2].split('=')[1].split(',')[0])
            gpu_time = float(parts[3].split('=')[1])

            file_sizes.append(size)
            cpu_times.append(cpu_time)
            gpu_times.append(gpu_time)

    plt.figure(figsize=(10, 6))
    plt.scatter(file_sizes, cpu_times, label='CPU', color='red')
    plt.scatter(file_sizes, gpu_times, label='GPU', color='blue')
    plt.xlabel('File Size (bytes)')
    plt.ylabel('Time (seconds)')
    plt.title('CPU vs GPU Time for Different File Sizes')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')  # Logarithmic scale for better visualization
    plt.yscale('log')  # Logarithmic scale for better visualization
    plt.show()


if __name__ == '__main__':
    plot_graph('data.txt')
