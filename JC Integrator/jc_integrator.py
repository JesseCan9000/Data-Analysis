import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
from mplcursors import cursor as mpl_cursor
import math
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.ndimage import gaussian_filter

def zoom(event, ax):
    ''' Utility function to allow for zooming into/out of matplotlib plot with mouse scroll wheel'''
    base_scale = 1.1
    xdata = event.xdata
    ydata = event.ydata

    if event.button == 'up':
        # Zoom in
        ax.set_xlim(xdata - (xdata - ax.get_xlim()[0]) / base_scale, xdata + (ax.get_xlim()[1] - xdata) / base_scale)
        ax.set_ylim(ydata - (ydata - ax.get_ylim()[0]) / base_scale, ydata + (ax.get_ylim()[1] - ydata) / base_scale)
    elif event.button == 'down':
        # Zoom out
        ax.set_xlim(xdata - (xdata - ax.get_xlim()[0]) * base_scale, xdata + (ax.get_xlim()[1] - xdata) * base_scale)
        ax.set_ylim(ydata - (ydata - ax.get_ylim()[0]) * base_scale, ydata + (ax.get_ylim()[1] - ydata) * base_scale)
    ax.figure.canvas.draw_idle()

def order_of_magnitude(number):
    ''' This function gets the order of magnitude of a number. Used in onclick() when calculating
    the nearest point in the data to where you double click in the plot. This function is used to ensure that
    there is roughly equal weighting for the x and y direcitons in finding the nearest data point to where you double click'''
    return math.floor(math.log10(abs(number)))

def main(squidstat_data_csv, label_peaks=False):
    '''Prompts the user to double click on a pair of points. Calculates the area of the polygon bounded by the
    line between those two points and the data points between the two points. For 29 vs Temperature (C) data
    this means that the area calculated is the charge accumulated in that time in coulombs (C)'''
    squidstat_data = pd.read_csv(squidstat_data_csv, comment='#', nrows=436)

    fig, ax = plt.subplots()
    ax.scatter(squidstat_data['Temperature (C)'], squidstat_data['29'], color='blue')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('29')
    plt.title('29 vs. Temperature (C)')

    peak_time_bounds = []
    peak_indices = []
    peak_count = 1
    
    def onclick(event):
        '''Handler for when the user double clicks in the matplotlib plot area. Finds the nearest data point in the provided data
        relative to where you click. Gets the current and time at that nearest index.'''
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if event.dblclick and (event.xdata is not None) and (event.ydata is not None) and (xlim[0] <= event.xdata <= xlim[1]) and (ylim[0] <= event.ydata <= ylim[1]):
            ix = event.xdata
            iy = event.ydata

            time_order = order_of_magnitude(squidstat_data['Temperature (C)'].median())
            current_order = order_of_magnitude(squidstat_data['29'].mean())
            distances = np.sqrt(((squidstat_data['Temperature (C)'].values - ix)/math.pow(10, time_order))**2 + ((squidstat_data['29'].values - iy)*math.pow(10, -1*current_order))**2)
            nearest_index = np.argmin(distances)
            
            nearest_time = squidstat_data.iloc[nearest_index]['Temperature (C)']
            nearest_current = squidstat_data.iloc[nearest_index]['29']

            peak_time_bounds.append(nearest_time)
            print(f"Clicked point: Time (s)={nearest_time}, 29={nearest_current}")
    
    def compute_peak_area(event):
        '''Draws a polygon bounded by the two points double clicked by the user and the data points between thsoe two bounds.
        Uses Shapely to calculate the area of this 29 vs Temperature (C) polygon and convert it into nC'''
        nonlocal peak_count

        # Check if there are an even number of peaks clicked. Even is required because exactly two points define the bounds of a peak in this code.
        if len(peak_time_bounds) % 2 != 0:
            raise ValueError("Compute Peak Areas: Odd number of points double clicked. Ensure there are an even number of points selected to detect peaks.")

        # Find the indices corresponding to the clicked points
        start_time = min(peak_time_bounds[-2], peak_time_bounds[-1]) 
        end_time = max(peak_time_bounds[-2], peak_time_bounds[-1])
        start_index = (squidstat_data['Temperature (C)'] - start_time).abs().idxmin()
        end_index = (squidstat_data['Temperature (C)'] - end_time).abs().idxmin()

        # Get the current values corresponding to the clicked points
        start_current = squidstat_data.loc[start_index, '29']
        end_current = squidstat_data.loc[end_index, '29']

        # Create a polygon representing the area between the line made from the double clicked points and the curve
        polygon_points = [(start_time, start_current)]
        for index, row in squidstat_data.iterrows():
            if start_time <= row['Temperature (C)'] <= end_time:
                polygon_points.append((row['Temperature (C)'], row['29']))
        polygon_points.append((end_time, end_current))
        polygon = Polygon(polygon_points, closed=True, color='red', alpha=0.5)

        # Calculate area based on polygon
        vertices = polygon.get_xy()
        shapely_polygon = ShapelyPolygon(vertices)
        polygon_area = shapely_polygon.area
        print(f"Peak Area Abs. Value: {polygon_area}") # Print the peak area

        # Extract the exterior coordinates of the polygon
        exterior_coords = shapely_polygon.exterior.coords.xy
        exterior_x, exterior_y = exterior_coords[0], exterior_coords[1]

        # Create a Polygon patch and render it on the plot
        polygon_patch = Polygon(xy=list(zip(exterior_x, exterior_y)), closed=True, color='red', alpha=0.5)
        ax.add_patch(polygon_patch)

        # Render peak label
        if label_peaks:
            mid_time = (start_time + end_time) / 2
            ax.text(mid_time, 1.5 * (start_current + end_current) / 2, f'Peak {peak_count} Area Abs. Value\n{polygon_area*math.pow(10,9):.2f} nC', horizontalalignment='center', verticalalignment='center', color='white', path_effects=[pe.withStroke(linewidth=4, foreground="black")])

        peak_count += 1

    def undo_click(event):
        '''Undo the most recent double click event. This function is called when the Undo Click button is clicked. Useful if you have a
        misclick/the nearest data point from onclick() is incorrect. Since there needs to be an even number of double clicks to calculate 
        the next peak area, click the undo button if you receive the "Odd number of points double clicked..." error.'''
        if peak_time_bounds:
            ix = peak_time_bounds.pop()
            print(f"Undo Click: Time={ix} removed.")

    def remove_peak(event):
        '''Removes the most recent peak and peak label upon clicking the Remove Peak button. Useful for keeping the screen
        clean as you study more peaks'''
        if ax.patches:
            ax.patches.pop()
        if ax.texts:
            ax.texts.pop()

    def subtract_baseline(event):
        '''Subtracts a linear baseline defined by the two nearest points clicked by the user from the data'''
        if len(peak_time_bounds) < 2:
            raise ValueError("Subtract Baseline: Less than two points clicked. Ensure there are at least two points selected.")

        # Define point1 and point2 using the last two points clicked
        point1 = (peak_time_bounds[-2], squidstat_data.loc[(squidstat_data['Temperature (C)'] - peak_time_bounds[-2]).abs().idxmin(), '29'])
        point2 = (peak_time_bounds[-1], squidstat_data.loc[(squidstat_data['Temperature (C)'] - peak_time_bounds[-1]).abs().idxmin(), '29'])
        point1_time = peak_time_bounds[-2]
        point2_time = peak_time_bounds[-1]
        point1_current = squidstat_data.loc[(squidstat_data['Temperature (C)'] - point1_time).abs().idxmin(), '29']
        point2_current = squidstat_data.loc[(squidstat_data['Temperature (C)'] - point2_time).abs().idxmin(), '29']

        # Calculate the baseline
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        intercept = point1[1] - slope * point1[0]
        baseline = slope * squidstat_data['Temperature (C)'] + intercept

        # Subtract the baseline from the original data
        squidstat_data['Baseline Corrected 29'] = squidstat_data['29'] - baseline
        yhat = gaussian_filter(squidstat_data['Baseline Corrected 29'], 6) # std dev for gaussian kernel = 6

        # Plot the baseline corrected data
        fig, ax = plt.subplots()
        ax.scatter(squidstat_data['Temperature (C)'], squidstat_data['Baseline Corrected 29'], label='Baseline Corrected Data', color="blue")
        ax.plot(squidstat_data['Temperature (C)'], yhat, label='Smoothed Baseline Corrected Data', color="red") # using Savitzky-Golay filter
        ax.set_xlabel('Temperature (C)')
        ax.set_ylabel('29 - Baseline')
        plt.title('Baseline Corrected Data')
        plt.legend()
        plt.show()

        print(f"Baseline subtracted using points: ({point1_time}, {point1_current}) and ({point2_time}, {point2_current})")

        # Find the maximum y value between point1 and point2
        index1 = np.searchsorted(squidstat_data['Temperature (C)'], point1[0])
        index2 = np.searchsorted(squidstat_data['Temperature (C)'], point2[0])
        max_y = np.max(yhat[index1:index2+1])
        max_y_x = squidstat_data['Temperature (C)'][np.argmax(yhat[index1:index2+1]) + index1]
        print(f"Maximum y value between baseline points: {max_y}")
        print(f"Corresponding x value: {max_y_x}")


    # Add buttons to the matplotlib figure and link zoom and onclick handlers
    undo_click_ax = plt.axes([0.88, 0.01, 0.1, 0.05])
    undo_click_button = Button(undo_click_ax, 'Undo Click')
    undo_click_button.on_clicked(undo_click)

    remove_peak_ax = plt.axes([0.13, 0.01, 0.2, 0.05])
    remove_peak_button = Button(remove_peak_ax, 'Remove Peak')
    remove_peak_button.on_clicked(remove_peak)
    
    compute_ax = plt.axes([0.625, 0.01, 0.2, 0.05])
    compute_button = Button(compute_ax, 'Compute Peak Area')
    compute_button.on_clicked(compute_peak_area)

    baseline_subtracted_ax = plt.axes([0.88, 0.50, 0.1, 0.05])
    baseline_subtracted_button = Button(baseline_subtracted_ax, 'Subtract\nBaseline')
    baseline_subtracted_button.on_clicked(subtract_baseline)

    fig.canvas.mpl_connect('scroll_event', lambda event: zoom(event, ax))
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == "__main__":
    # USER: MODIFY THESE LINES TO BE THE FILE PATH OF THE CORRESPODING DATA FILES
    # squidstat_data_csv = "C:/Users/jcmar/OneDrive/Desktop/Data/CO_TPD/05212024/TPD 1 2024-05-21_16.56.21_1.csv"
    squidstat_data_csv = "C:/Users/jcmar/OneDrive/Desktop/Data/CO_TPD/05212024/TPD 2 2024-05-21_18.26.14_1.csv"
    # squidstat_data_csv = "C:/Users/jcmar/OneDrive/Desktop/Data/CO_TPD/05222024/TPD 1 2024-05-22_11.52.27_1.csv"
    # squidstat_data_csv = "C:/Users/jcmar/OneDrive/Desktop/Data/CO_TPD/05222024/TPD 2 2024-05-22_15.41.55_1.csv"
    # squidstat_data_csv = "C:/Users/jcmar/OneDrive/Desktop/Data/CO_TPD/05232024/TPD 1 2024-05-23_16.41.39_1.csv"

    DEVICE_AREA = 1.1 # cm2

    # squidstat_data_csv = '150C trial 3.csv'
    # squidstat_data_csv = 'test.csv'

    # plot_current_flow_rate_and_temperature(squidstat_data_csv, mfc_data_csv, temperature_data_csv)

    main(squidstat_data_csv, label_peaks=True)