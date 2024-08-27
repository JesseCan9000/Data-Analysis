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

def main(dataframe_csv, label_peaks=False):
    ''' Prompts the user to double click on a pair of points. Calculates the area of the polygon bounded by the
    line between those two points and the data points between the two points. For 29 vs Temperature (C) data
    this means that the area calculated is the charge accumulated in that time in coulombs (C) '''
    dataframe = pd.read_csv(dataframe_csv, comment='#', nrows=nrows)

    dataframe = dataframe.iloc[:-1] # remove last row of data because it often has NaN for the mass of interest, which ruins this function

    dataframe[y_axis] = dataframe[y_axis]/CEM_gain

    fig, ax = plt.subplots()
    ax.scatter(dataframe[x_axis_column], dataframe[y_axis], color='blue')
    ax.set_xlabel(x_axis_column)
    ax.set_ylabel(y_axis)
    plt.title(f'29 vs. {x_axis_column}')

    peak_x_bounds = []
    peak_indices = []
    peak_count = 1
    
    def onclick(event):
        ''' Handler for when the user double clicks in the matplotlib plot area. Finds the nearest data point in the provided data
        relative to where you click. Gets the y and x at that nearest index. '''
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if event.dblclick and (event.xdata is not None) and (event.ydata is not None) and (xlim[0] <= event.xdata <= xlim[1]) and (ylim[0] <= event.ydata <= ylim[1]):
            ix = event.xdata
            iy = event.ydata

            x_order = order_of_magnitude(dataframe[x_axis_column].median())
            y_order = order_of_magnitude(dataframe[y_axis].mean())
            distances = np.sqrt(((dataframe[x_axis_column].values - ix)/math.pow(10, x_order))**2 + ((dataframe[y_axis].values - iy)/math.pow(10, y_order))**2)
            nearest_index = np.argmin(distances)
            
            nearest_x = dataframe.iloc[nearest_index][x_axis_column]
            nearest_y = dataframe.iloc[nearest_index][y_axis]

            peak_x_bounds.append(nearest_x)
            print(f"Clicked point: {x_axis_column}={nearest_x}, 29={nearest_y:.2e}")
    
    def compute_peak_area(event=None, convert_to_nC=True):
        ''' Draws a polygon bounded by the two points double clicked by the user and the data points between thsoe two bounds.
        Uses Shapely to calculate the area of this 29 vs Temperature (C) polygon and convert it into nC '''
        nonlocal peak_count

        # Check if there are an even number of peaks clicked. Even is required because exactly two points define the bounds of a peak in this code.
        if len(peak_x_bounds) % 2 != 0:
            raise ValueError("Compute Peak Areas: Odd number of points double clicked. Ensure there are an even number of points selected to detect peaks.")

        # Find the indices corresponding to the clicked points
        start_x = min(peak_x_bounds[-2], peak_x_bounds[-1]) 
        end_x = max(peak_x_bounds[-2], peak_x_bounds[-1])
        start_index = (dataframe[x_axis_column] - start_x).abs().idxmin()
        end_index = (dataframe[x_axis_column] - end_x).abs().idxmin()

        # Get the y values corresponding to the clicked points
        start_y = dataframe.loc[start_index, y_axis]
        end_y = dataframe.loc[end_index, y_axis]

        # Create a polygon representing the area between the line made from the double clicked points and the curve
        polygon_points = [(start_x, start_y)]
        for index, row in dataframe.iterrows():
            if start_x <= row[x_axis_column] <= end_x:
                polygon_points.append((row[x_axis_column], row[y_axis]))
        polygon_points.append((end_x, end_y))
        polygon = Polygon(polygon_points, closed=True, color='red', alpha=0.5)

        # Calculate area based on polygon
        vertices = polygon.get_xy()
        shapely_polygon = ShapelyPolygon(vertices)
        polygon_area = shapely_polygon.area
        print(f"Peak Area Abs. Value: {polygon_area:.2e}") # Print the peak area

        # Extract the exterior coordinates of the polygon
        exterior_coords = shapely_polygon.exterior.coords.xy
        exterior_x, exterior_y = exterior_coords[0], exterior_coords[1]

        # Create a Polygon patch and render it on the plot
        polygon_patch = Polygon(xy=list(zip(exterior_x, exterior_y)), closed=True, color='red', alpha=0.5)
        ax.add_patch(polygon_patch)

        # Render peak label
        if label_peaks:
            mid_x = (start_x + end_x) / 2

            if convert_to_nC:
                ax.text(mid_x, 1.5 * (start_y + end_y) / 2, f'Peak {peak_count} Area Abs. Value\n{polygon_area*math.pow(10,9):.2e} nC', horizontalalignment='center', verticalalignment='center', color='white', path_effects=[pe.withStroke(linewidth=4, foreground="black")])
            else:
                ax.text(mid_x, 1.5 * (start_y + end_y) / 2, f'Peak {peak_count} Area Abs. Value\n{polygon_area:.2e} {y_axis_units}•{x_axis_units}', horizontalalignment='center', verticalalignment='center', color='white', path_effects=[pe.withStroke(linewidth=4, foreground="black")])

        peak_count += 1

        return polygon_area, start_index, end_index, start_x, end_x

    def undo_click(event):
        ''' Undo the most recent double click event. This function is called when the Undo Click button is clicked. Useful if you have a
        misclick/the nearest data point from onclick() is incorrect. Since there needs to be an even number of double clicks to calculate 
        the next peak area, click the undo button if you receive the "Odd number of points double clicked..." error. '''
        if peak_x_bounds:
            ix = peak_x_bounds.pop()
            print(f"Undo Click: Time={ix} removed.")

    def remove_peak(event):
        '''Removes the most recent peak and peak label upon clicking the Remove Peak button. Useful for keeping the screen
        clean as you study more peaks'''
        if ax.patches:
            ax.patches.pop()
        if ax.texts:
            ax.texts.pop()

    # def leading_edge(event):
    #     if len(peak_x_bounds) < 2:
    #         raise ValueError("Subtract Baseline: Less than two points clicked. Ensure there are at least two points selected.")

    #     # Define point1 and point2 using the last two points clicked
    #     point1 = (peak_x_bounds[-2], dataframe.loc[(dataframe[x_axis_column] - peak_x_bounds[-2]).abs().idxmin(), y_axis])
    #     point2 = (peak_x_bounds[-1], dataframe.loc[(dataframe[x_axis_column] - peak_x_bounds[-1]).abs().idxmin(), y_axis])
    #     point1_x = peak_x_bounds[-2]
    #     point2_x = peak_x_bounds[-1]
    #     point1_y = dataframe.loc[(dataframe[x_axis_column] - point1_x).abs().idxmin(), y_axis]
    #     point2_y = dataframe.loc[(dataframe[x_axis_column] - point2_x).abs().idxmin(), y_axis]

    #     # Calculate E, the activation energy of desorption from slope of ln(p) vs 1/T [=] 1/K
    #     slope = (np.log(point2[1]) - np.log(point1[1])) / ( (1/(point2[0]+273)) - (1/point1[0]+273) )
    #     #intercept = point1[1] - slope * point1[0]
        
    #     R = 0.008314 # kJ/mol•K
    #     E = -1*slope*R

    #     print(f'Activation energy of desorption E = {E} kJ/mol')

    def subtract_baseline(event):
        ''' Subtracts a linear baseline defined by the two nearest points clicked by the user from the data '''
        if len(peak_x_bounds) < 2:
            raise ValueError("Subtract Baseline: Less than two points clicked. Ensure there are at least two points selected.")

        # Define point1 and point2 using the last two points clicked
        point1 = (peak_x_bounds[-2], dataframe.loc[(dataframe[x_axis_column] - peak_x_bounds[-2]).abs().idxmin(), y_axis])
        point2 = (peak_x_bounds[-1], dataframe.loc[(dataframe[x_axis_column] - peak_x_bounds[-1]).abs().idxmin(), y_axis])
        point1_x = peak_x_bounds[-2]
        point2_x = peak_x_bounds[-1]
        point1_y = dataframe.loc[(dataframe[x_axis_column] - point1_x).abs().idxmin(), y_axis]
        point2_y = dataframe.loc[(dataframe[x_axis_column] - point2_x).abs().idxmin(), y_axis]

        # Calculate the baseline
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        intercept = point1[1] - slope * point1[0]
        baseline = slope * dataframe[x_axis_column] + intercept

        # Subtract the baseline from the original data
        dataframe['Baseline Corrected 29'] = dataframe[y_axis] - baseline
        yhat = gaussian_filter(dataframe['Baseline Corrected 29'], 6) # std dev for gaussian kernel = 6

        # Plot the baseline corrected data
        fig, ax = plt.subplots()
        ax.scatter(dataframe[x_axis_column], dataframe['Baseline Corrected 29'], label='Baseline Corrected Data', color="blue")
        ax.plot(dataframe[x_axis_column], yhat, label='Smoothed Baseline Corrected Data', color="red") # using Savitzky-Golay filter
        ax.set_xlabel(x_axis_column)
        ax.set_ylabel('29 - Baseline')
        plt.title('Baseline Corrected Data')
        plt.legend()

        # dataframe.to_csv('out.csv', index=False)
        plt.show()

        print(f"Baseline subtracted using points: ({point1_x}, {point1_y:.2e}) and ({point2_x}, {point2_y:.2e})")

        # Find the maximum y value between point1 and point2
        index1 = np.searchsorted(dataframe[x_axis_column], point1[0])
        index2 = np.searchsorted(dataframe[x_axis_column], point2[0])
        max_y = np.max(yhat[index1:index2+1])
        max_y_x = dataframe[x_axis_column][np.argmax(yhat[index1:index2+1]) + index1]
        print(f"Maximum y value between baseline points: {max_y:.2e}")
        print(f"Corresponding x value: {max_y_x}")

    def calculate_mols_desorbed(event):
        ''' Calculates the moles desorbed using the peak area of P vs time, reactor volume, R, and T_avg. Only valid in versus_time = True mode '''

        R = 62.36 # (L•torr)/(mol•K)

        integral, start_index, end_index, start_x, end_x = compute_peak_area(convert_to_nC=False) # integral of y(x)dx

        print()
        print(f'Integral: {integral:.2e} {y_axis_units}•{x_axis_units}')

        print(f'Reactor Volume: {reactor_volume} L')

        subset = dataframe['Temperature (C)'][start_index:end_index]
        T_avg = subset.mean() + 273

        print(f'Average T: {T_avg:.2f} K')

        delta_t = end_x - start_x

        print(f'Delta t: {delta_t:.2f}')

        n_tilde = (integral*reactor_volume)/(R*T_avg) # mols•s
        n = n_tilde/delta_t # mols of gas desorbed

        print(f'Moles of gas desorbed: {n:.2e} mols')
        print()

    # Add buttons to the matplotlib figure and link zoom and onclick handlers
    undo_click_ax = plt.axes([0.88, 0.01, 0.13, 0.05])
    undo_click_button = Button(undo_click_ax, 'Undo Click')
    undo_click_button.on_clicked(undo_click)

    remove_peak_ax = plt.axes([0.13, 0.01, 0.17, 0.05])
    remove_peak_button = Button(remove_peak_ax, 'Remove Peak')
    remove_peak_button.on_clicked(remove_peak)
    
    compute_ax = plt.axes([0.625, 0.01, 0.23, 0.05])
    compute_button = Button(compute_ax, 'Compute Peak Area')
    compute_button.on_clicked(compute_peak_area)

    subtract_baseline_ax = plt.axes([0.88, 0.50, 0.12, 0.07])
    subtract_baseline_button = Button(subtract_baseline_ax, 'Subtract\nBaseline')
    subtract_baseline_button.on_clicked(subtract_baseline)

    if versus_time == True:
        mols_desorbed_ax = plt.axes([0.88, 0.25, 0.12, 0.07])
        mols_desorbed_button = Button(mols_desorbed_ax, 'Moles\nDesorbed')
        mols_desorbed_button.on_clicked(calculate_mols_desorbed)

    # leading_edge_ax = plt.axes([0.88, 0.25, 0.1, 0.05])
    # leading_edge_button = Button(leading_edge_ax, 'Leading\nEdge')
    # leading_edge_button.on_clicked(leading_edge)

    fig.canvas.mpl_connect('scroll_event', lambda event: zoom(event, ax))
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == "__main__":
    # USER: MODIFY THESE LINES TO BE THE FILE PATH OF THE CORRESPODING DATA FILES AND OPTIONS FOR THE SCRIPT

    dataframe_csv = "C:/Users/jcmar/OneDrive/Desktop/Data/CO_TPD/Device CNT 2/08232024/13CO TPD 1 2024-08-23_12.05.55_1.csv" 

    DEVICE_AREA = 1 # cm2
    reactor_volume = 4.397 # L
    CEM_gain = 1064

    versus_time = False # Plot versus time if True, plot versus temperature if False

    y_axis = '29' # Change to the column in your csv file you would like to use as the y axis. For example: '29' for looking at the mass 29 from a mass spec signal
    y_axis_units = 'torr'

    ###############################################################################################################

    if versus_time:
        x_axis_column = 'Time (s)'
        x_axis_units = 's'
    else:
        x_axis_column = 'Temperature (C)'
        x_axis_units = 'C'

    nrows = 600

    main(dataframe_csv, label_peaks=True)