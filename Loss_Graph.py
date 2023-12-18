# /Users/mac/Desktop/11월16일까지 어케든 완성시켜/지표/PGGANtraining_metrics2.csv, Average
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file
latest_file_path = '/Users/mac/Desktop/11월16일까지 어케든 완성시켜/지표/PGGANtraining_metrics2.csv'
latest_data = pd.read_csv(latest_file_path)

# Trimming the data to 1000 epochs
latest_data_trimmed = latest_data[latest_data['Epoch'] < 1000]

# Extracting data
latest_epochs = latest_data_trimmed['Epoch']
generator_loss = latest_data_trimmed['Generator Loss']
discriminator_loss = latest_data_trimmed['Discriminator Loss']

# Creating a larger plot
plt.figure(figsize=(15, 8))

# Plotting the data with markers every 20 epochs
plt.plot(latest_epochs, generator_loss, marker='o', linestyle='-', color='g', label=' Average Generator Loss', markevery=20)
plt.plot(latest_epochs, discriminator_loss,  linestyle='-', color='m', label=' Average Discriminator Loss', markevery=20)

# Setting the title and labels
# plt.title('Generator and Discriminator Loss Over Epochs (Up to 1000 Epochs)', fontsize=14)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Average Loss', fontsize=20)

# Customizing the x-axis and y-axis tick labels for better readability
plt.xticks(range(0, 1001, 100), fontsize=20)
plt.yticks(fontsize=20)

# Adding a legend and grid
plt.legend(fontsize=20)
plt.grid(True)

# Displaying the plot
plt.show()
