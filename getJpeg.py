import os
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image

# Specify the folder path
output_folder = 'output_images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read CSV data into a pandas DataFrame
df = pd.read_csv('ohcl_cnn.csv')
df_cross = pd.read_csv('df_buy.csv')


# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], unit='s')
df_cross['time'] = pd.to_datetime(df_cross['time'], unit='s')

# Set the 'time' column as the index
df.set_index('time', inplace=True)
df_cross.set_index('time', inplace=True)

# Sort the DataFrame by time
df = df.sort_index()
df_cross = df_cross.sort_index()

max_length = 100

for i in range(max_length, len(df) + 1):
    # Extract the current 200 bars
    current_bars = df.iloc[i - max_length:i]

    # Check if the current time is in df_cross
    if current_bars.index[-1] in df_cross.index:
        # Plot the OHLC data using mplfinance
        mpf.plot(current_bars, type='candle',
                 savefig=os.path.join(output_folder, f"{int(current_bars.index[-1].timestamp())}.png"),
                 ylabel='Price', title=f"OHLC Chart - {current_bars.index[-1]}",
                 style='yahoo', figscale=1.5)  # Adjust the figscale value as needed

        # Close the plot to prevent overlapping
        plt.close()

        # Open the saved image and crop
        image_path = os.path.join(output_folder, f"{int(current_bars.index[-1].timestamp())}.png")
        image = Image.open(image_path)
        cropped_image = image.crop((210, 100, image.width - 120, image.height - 160))

        # Save the cropped image
        cropped_image.save(image_path)