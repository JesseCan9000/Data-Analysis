import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
from matplotlib.text import Text
from mplcursors import cursor as mpl_cursor
import math
import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

custom_rc_params = {
    'axes.labelsize': 28,   # Font size for x and y labels
    'xtick.labelsize': 22,  # Font size for x-axis tick labels
    'ytick.labelsize': 22,  # Font size for y-axis tick labels
}

plt.rcParams.update(custom_rc_params)

class ToolTip:
    """Class to create a tooltip for a given widget."""
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        # Bind events to show and hide the tooltip
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        """Display the tooltip near the widget."""
        # Create a new top-level window for the tooltip
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{self.widget.winfo_rootx() + 20}+{self.widget.winfo_rooty() + 20}")
        
        # Create a label inside the tooltip window
        label = tk.Label(self.tooltip_window, text=self.text, background="light yellow", relief='solid', borderwidth=2)
        label.pack(ipadx=5, ipady=5)

    def hide_tooltip(self, event=None):
        """Hide the tooltip if it's visible."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert('end', message)
        self.text_widget.see('end')

# # Work in progress DraggableText class
# class DraggableText:
#     def __init__(self, text):
#         self.text = text
#         self.press = None
#         self.text.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.text.figure.canvas.mpl_connect('button_release_event', self.on_release)
#         self.text.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

#     def on_press(self, event):
#         if event.inaxes != self.text.axes:
#             return
#         contains, attr = self.text.contains(event)
#         if not contains:
#             return
#         # Store the mouse click and the original position
#         self.press = (self.text.get_position(), event.xdata, event. ydata)

#     def on_motion(self, event):
#         if self.press is None or event.inaxes != self.text.axes:
#             return
#         # Calculate the new position of the text
#         position, xpress, ypress = self.press
#         dx = event.xdata - xpress
#         dy = event.ydata - ypress
#         new_position = (position[0] + dx, position[1] + dy)
#         self.text.set_position(new_position)
#         self.text.figure.canvas.draw()

#     def on_release(self, event):
#         self.press = None
#         self.text.figure.canvas.draw()

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

def main(dataframe, label_peaks=False):
    ''' Prompts the user to double click on a pair of points. Calculates the area of the polygon bounded by the
    line between those two points and the data points between the two points. For 29 vs Temperature (C) data
    this means that the area calculated is the charge accumulated in that time in coulombs (C) '''

    ############## DELETE THIS LINE WHEN DONE TESTING ##############
    dataframe = dataframe.iloc[:-1] # remove last row of data because it often has NaN for the mass of interest, which ruins this function

    x_label = x_col_name.get()
    y_label = y_col_name.get()

    fig, ax = plt.subplots()
    ax.scatter(dataframe[x_label], dataframe[y_label], color='blue')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_position([0.24, 0.25, 0.67, 0.67])
    # plt.title(f'29 vs. {x_label}')

    peak_x_bounds = []
    peak_y_bounds = []
    peak_indices = []
    peak_count = 1
    
    def onclick(event):
        ''' Handler for when the user double clicks in the matplotlib plot area. Finds the nearest data point in the provided data
        relative to where you click. Gets the y and x at that nearest index. '''
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if event.dblclick and event.inaxes and not event.key == 'control':
            # Ignore double-clicks on text objects
            for artist in event.inaxes.get_children():
                if isinstance(artist, Text) and artist.contains(event)[0]:
                    return

            ix = event.xdata
            iy = event.ydata

            x_order = order_of_magnitude(dataframe[x_label].median())
            y_order = order_of_magnitude(dataframe[y_label].median())
            distances = np.sqrt(((dataframe[x_label].values - ix)/math.pow(10, x_order))**2 + ((dataframe[y_label].values - iy)/math.pow(10, y_order))**2)
            nearest_index = np.argmin(distances)
            
            nearest_x = dataframe.iloc[nearest_index][x_label]
            nearest_y = dataframe.iloc[nearest_index][y_label]

            peak_x_bounds.append(nearest_x)
            peak_y_bounds.append(nearest_y)
            print(f"• Clicked point: {x_label}={nearest_x}, {y_label}={nearest_y:.2e}")
    
    def calculate_fwhm(dataframe, start_index, end_index, x_label, y_label, x_axis_units):
        # Step 1: Calculate the half-max value
        min_y = min(dataframe.loc[start_index:end_index, y_label])
        max_y = max(dataframe.loc[start_index:end_index, y_label])
        half_max = (max_y - min_y) / 2 + min_y
        # print(f'• Half Max: {half_max}')

        # Step 2: Prepare data
        peak_data = dataframe.loc[start_index:end_index, [x_label, y_label]]
        x_values = peak_data[x_label].values
        y_values = peak_data[y_label].values

        # Create cubic spline interpolation function
        interp = interp1d(x_values, y_values, kind='cubic', bounds_error=False, fill_value="extrapolate")

        try:
            # Find all points where y is above or below the half_max
            above_half_max = y_values >= half_max
            
            # Find the transitions
            transitions = np.where(np.diff(above_half_max))[0]
            
            if len(transitions) < 2:
                raise ValueError("Not enough transitions found around half maximum")
            
            # Find the actual interpolated x values at half max
            interpolated_x = []
            for idx in transitions:
                # Linear interpolation between the two points around the transition
                x1, x2 = x_values[idx], x_values[idx+1]
                y1, y2 = y_values[idx], y_values[idx+1]
                
                # Interpolate to find x at half_max
                x_at_half_max = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                interpolated_x.append(x_at_half_max)
            
            # Ensure we have two different points
            if len(set(interpolated_x)) < 2:
                raise ValueError("Could not find distinct half-max points")
            
            # Take the leftmost and rightmost points if multiple found
            lower_x = min(interpolated_x)
            upper_x = max(interpolated_x)
            
            # print(f'• Interpolated X (Lower, Upper): ({lower_x}, {upper_x})')
            
            # Calculate FWHM
            fwhm = abs(upper_x - lower_x)
            print(f'• FWHM: {fwhm:.2e} {x_axis_units}')

            ax.hlines(half_max, xmin=lower_x, xmax=upper_x, color='green', linestyle='-')
            
            return fwhm
            
        except ValueError as e:
            print(f"Error in FWHM calculation: {e}")
            return None

    def compute_peak_area(event=None, convert_to_nC=True):
        ''' Draws a polygon bounded by the two points double clicked by the user and the data points between thsoe two bounds.
        Uses Shapely to calculate the area of this 29 vs Temperature (C) polygon and convert it into nC '''
        nonlocal peak_count

        x_axis_units = x_units.get()
        y_axis_units = y_units.get()

        # Check if there are an even number of peaks clicked. Even is required because exactly two points define the bounds of a peak in this code.
        if len(peak_x_bounds) % 2 != 0:
            raise ValueError("Compute Peak Areas: Odd number of points double clicked. Ensure there are an even number of points selected to detect peaks.")

        # Find the indices corresponding to the clicked points
        start_x = min(peak_x_bounds[-2], peak_x_bounds[-1]) 
        end_x = max(peak_x_bounds[-2], peak_x_bounds[-1])
        start_index = (dataframe[x_label] - start_x).abs().idxmin()
        end_index = (dataframe[x_label] - end_x).abs().idxmin()

        # Get the y values corresponding to the clicked points
        start_y = dataframe.loc[start_index, y_label]
        end_y = dataframe.loc[end_index, y_label]

        # Create a polygon representing the area between the line made from the double clicked points and the curve
        polygon_points = [(start_x, start_y)]
        for index, row in dataframe.iterrows():
            if (start_x <= row[x_label] <= end_x): # and (min(start_y, end_y) <= row[y_label]):
                polygon_points.append((row[x_label], row[y_label]))
        polygon_points.append((end_x, end_y))
        polygon = Polygon(polygon_points, closed=True, color='red', alpha=0.5)
        # Calculate area based on polygon
        vertices = polygon.get_xy()
        shapely_polygon = ShapelyPolygon(vertices)
        polygon_area = shapely_polygon.area

        # Print the peak area
        if x_axis_units and y_axis_units:
            print(f"• Peak Area Abs. Value: {polygon_area:.3e} {y_axis_units}•{x_axis_units}")
        else:
            print(f"• Peak Area Abs. Value: {polygon_area:.3e}")
        # Extract the exterior coordinates of the polygon
        exterior_coords = shapely_polygon.exterior.coords.xy
        exterior_x, exterior_y = exterior_coords[0], exterior_coords[1]

        # Print the triangle area correction (only applicable to adsorption peaks)
        base = abs(start_x - end_x)
        height = abs(start_y - end_y)
        area = 0.5*base*height
        print(f"• Triangle Correction Area: {area:.3e}")

        # # Calculate Full Width at Half Maximum (FWHM)
        calculate_fwhm(dataframe, start_index, end_index, x_label, y_label, x_axis_units)
        # half_max = (max(dataframe.loc[start_index:end_index, y_label]) - min(dataframe.loc[start_index:end_index, y_label]))/2 + min(dataframe.loc[start_index:end_index, y_label])
        # print(f'• Half Max: {half_max}')



        # left_idx = (dataframe.loc[start_index:end_index, y_label] - half_max).abs().idxmin()
        # print(f'• Left Index: {left_idx}')
        # right_idx = (dataframe.loc[left_idx:end_index, y_label] - half_max).abs().idxmin()
        # print(f'• Right Index: {right_idx}')

        # x_order = order_of_magnitude(dataframe[x_label].median())
        # y_order = order_of_magnitude(dataframe[y_label].median())
        # distances = np.sqrt(((dataframe[x_label].values - left_idx)/math.pow(10, x_order))**2 + ((dataframe[y_label].values - iy)/math.pow(10, y_order))**2)
        # nearest_index = np.argmin(distances)
        
        # nearest_x = dataframe.iloc[nearest_index][x_label]
        # nearest_y = dataframe.iloc[nearest_index][y_label]

        # fwhm = dataframe.loc[right_idx, x_label] - dataframe.loc[left_idx, x_label]
        # print(f"• FWHM: {fwhm:.2e} {x_axis_units}")

        # Create a Polygon patch and render it on the plot
        polygon_patch = Polygon(xy=list(zip(exterior_x, exterior_y)), closed=True, color='red', alpha=0.5)
        ax.add_patch(polygon_patch)

        # Render peak label
        if label_peaks:
            mid_x = (start_x + end_x) / 2

            if convert_to_nC:
                ax.text(mid_x, 1.5 * (start_y + end_y) / 2, f'Peak {peak_count} Area Abs. Value\n{polygon_area*math.pow(10,9):.3e} nC', horizontalalignment='center', verticalalignment='center', color='white', path_effects=[pe.withStroke(linewidth=4, foreground="black")])
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
            iy = peak_y_bounds.pop()
            print(f"• Undo Click: {x_label}={ix}, {y_label}={iy} removed.")

    def remove_peak(event):
        '''Removes the most recent peak and peak label upon clicking the Remove Peak button. Useful for keeping the screen
        clean as you study more peaks'''
        if ax.patches:
            ax.patches.pop()
        if ax.texts:
            ax.texts.pop()
        # if ax.hlines:
        #     ax.hlines.pop()

    # def leading_edge(event):
    #     if len(peak_x_bounds) < 2:
    #         raise ValueError("Subtract Baseline: Less than two points clicked. Ensure there are at least two points selected.")

    #     # Define point1 and point2 using the last two points clicked
    #     point1 = (peak_x_bounds[-2], dataframe.loc[(dataframe[x_label] - peak_x_bounds[-2]).abs().idxmin(), y_label])
    #     point2 = (peak_x_bounds[-1], dataframe.loc[(dataframe[x_label] - peak_x_bounds[-1]).abs().idxmin(), y_label])
    #     point1_x = peak_x_bounds[-2]
    #     point2_x = peak_x_bounds[-1]
    #     point1_y = dataframe.loc[(dataframe[x_label] - point1_x).abs().idxmin(), y_label]
    #     point2_y = dataframe.loc[(dataframe[x_label] - point2_x).abs().idxmin(), y_label]

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
        point1 = (peak_x_bounds[-2], dataframe.loc[(dataframe[x_label] - peak_x_bounds[-2]).abs().idxmin(), y_label])
        point2 = (peak_x_bounds[-1], dataframe.loc[(dataframe[x_label] - peak_x_bounds[-1]).abs().idxmin(), y_label])
        point1_x = peak_x_bounds[-2]
        point2_x = peak_x_bounds[-1]
        point1_y = dataframe.loc[(dataframe[x_label] - point1_x).abs().idxmin(), y_label]
        point2_y = dataframe.loc[(dataframe[x_label] - point2_x).abs().idxmin(), y_label]

        # Calculate the baseline
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        intercept = point1[1] - slope * point1[0]
        baseline = slope * dataframe[x_label] + intercept

        # Subtract the baseline from the original data
        dataframe['Baseline Corrected 29'] = dataframe[y_label] - baseline
        yhat = gaussian_filter(dataframe['Baseline Corrected 29'], gaussian_filter_slider.val)

        # Plot the baseline corrected data
        fig, ax = plt.subplots()
        ax.scatter(dataframe[x_label], dataframe['Baseline Corrected 29'], label='Baseline Corrected Data', color="blue")
        ax.plot(dataframe[x_label], yhat, label='Smoothed Baseline Corrected Data', color="red") # using Savitzky-Golay filter
        ax.set_xlabel(x_label)
        ax.set_ylabel('29 - Baseline')
        plt.title('Baseline Corrected Data')
        plt.legend()

        # dataframe.to_csv('out.csv', index=False)
        plt.show()

        print(f"• Baseline subtracted using points: ({point1_x}, {point1_y:.2e}) and ({point2_x}, {point2_y:.2e}), and gaussian filter stdev: {gaussian_filter_slider.val:.2f}")

        # Find the maximum y value between point1 and point2
        index1 = np.searchsorted(dataframe[x_label], point1[0])
        index2 = np.searchsorted(dataframe[x_label], point2[0])
        max_y = np.max(yhat[index1:index2+1])
        max_y_x = dataframe[x_label][np.argmax(yhat[index1:index2+1]) + index1]
        print(f"• Maximum y value between baseline points: {max_y:.2e}")
        print(f"• Corresponding x value: {max_y_x}")

    def calculate_mols_desorbed(event):
        ''' Calculates the moles desorbed using the peak area of P vs time, reactor volume, R, and T_avg. Only valid in versus_time = True mode '''

        R = 62.36 # (L•torr)/(mol•K)

        integral, start_index, end_index, start_x, end_x = compute_peak_area(convert_to_nC=False) # integral of y(x)dx

        print()
        print(f'• Integral: {integral:.2e} {y_axis_units}•{x_axis_units}')

        print(f'• Reactor Volume: {reactor_volume} L')

        subset = dataframe['Temperature (C)'][start_index:end_index]
        T_avg = subset.mean() + 273

        print(f'• Average T: {T_avg:.2f} K')

        delta_t = end_x - start_x

        print(f'• Delta t: {delta_t:.2f}')

        n_tilde = (integral*reactor_volume)/(R*T_avg) # mols•s
        n = n_tilde/delta_t # mols of gas desorbed

        print(f'• Moles of gas desorbed: {n:.2e} mols')
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

    # gaussian_filter_ax = plt.axes([0.92, 0.60, 0.07, 0.15])
    # gaussian_filter_slider = Slider(gaussian_filter_ax, 'Gaussian Filter\nStd Dev', 0, 10, valinit=5, orientation='vertical')

    # subtract_baseline_ax = plt.axes([0.92, 0.50, 0.07, 0.07])
    # subtract_baseline_button = Button(subtract_baseline_ax, 'Subtract\nBaseline')
    # subtract_baseline_button.on_clicked(subtract_baseline)

    # if versus_time == True:
    #     mols_desorbed_ax = plt.axes([0.88, 0.25, 0.12, 0.07])
    #     mols_desorbed_button = Button(mols_desorbed_ax, 'Moles\nDesorbed')
    #     mols_desorbed_button.on_clicked(calculate_mols_desorbed)

    # leading_edge_ax = plt.axes([0.88, 0.25, 0.1, 0.05])
    # leading_edge_button = Button(leading_edge_ax, 'Leading\nEdge')
    # leading_edge_button.on_clicked(leading_edge)

    fig.canvas.mpl_connect('scroll_event', lambda event: zoom(event, ax))
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == "__main__":
    # Create a Tkinter root window
    root = tk.Tk()
    root.title('JC Integrator')

    # This function runs faster if placed after initializing tk window root than before it
    def get_path():
        # global dataframe
        
        csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])  # Get the path from file dialog
        print(f'File name: {csv_path.split("/")[-1]}')

        if csv_path == '':
            csv_path == 'Please choose a CSV file'
            path_label.config(text=csv_path)  # Update the path label text

            return
        else:
            path_label.config(text=csv_path)
            dataframe = pd.read_csv(csv_path, comment='#')  # Load the dataframe

            # Update the column names in the Comboboxes
            y_col_name['values'] = list(dataframe.columns)
            x_col_name['values'] = list(dataframe.columns)

            # Submit button
            submit_btn = ttk.Button(root, text="Submit", command=lambda:main(dataframe))
            submit_btn.grid(row=5, column=1, padx=10, pady=10)

    # global dataframe
    dataframe = pd.DataFrame(['N/A'], columns=['Please choose a CSV file'])  # Initialize empty dataframe

    # Y-axis column name
    ttk.Label(root, text="Y-axis:").grid(row=0, column=0, padx=10, pady=5)
    y_col_name = ttk.Combobox(root, values=list(dataframe.columns))
    y_col_name.grid(row=0, column=1, padx=10, pady=5)

    # Y-axis units
    ttk.Label(root, text="Y-axis Units:").grid(row=1, column=0, padx=10, pady=5)
    y_units = ttk.Entry(root)
    y_units.grid(row=1, column=1, padx=10, pady=5)

    # X-axis column name
    ttk.Label(root, text="X-axis:").grid(row=2, column=0, padx=10, pady=5)
    x_col_name = ttk.Combobox(root, values=list(dataframe.columns))
    x_col_name.grid(row=2, column=1, padx=10, pady=5)

    # X-axis units
    ttk.Label(root, text="X-axis Units:").grid(row=3, column=0, padx=10, pady=5)
    x_units = ttk.Entry(root)
    x_units.grid(row=3, column=1, padx=10, pady=5)

    # File path selection
    ttk.Label(root, text='CSV File:').grid(row=4, column=0, padx=10, pady=5)
    path_label = ttk.Label(root, text='Please choose a CSV file')
    path_label.grid(row=4, column=1, padx=10, pady=5)

    # Choose file button
    choose_file_btn = ttk.Button(root, text="Choose File", command=get_path)
    choose_file_btn.grid(row=4, column=2, padx=10, pady=5)

    # Create a question mark label in the top-right corner
    question_mark = ttk.Label(root, text="?", foreground="grey", font=("Arial", 14, "bold"))
    question_mark.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)  # Position in top-right corner

    # Add a tooltip to the question mark
    instructions = """Instructions:
    1. Click "Choose File" to select a CSV file
    2. Choose the appropriate X and Y axis columns from the dropdowns
    3. (Optional) Enter the units for both axes
    4. Submit the selections to generate the plot
    5. Double click to select points
    6. Subtract baseline or calculate peak area as needed
    
    Choosing a file can sometimes make the program freeze. Just wait a few seconds and it should work."""
    ToolTip(question_mark, text=instructions)

    # Add a text box at the bottom of the window for logging print statements
    log_text = tk.Text(root, height=10, width=50)
    log_text.grid(row=6, columnspan=5, padx=10, pady=10)


    # Replace the sys.stdout with our own PrintLogger
    sys.stdout = PrintLogger(log_text)

    # Run the application
    root.mainloop()



#### TODO:
# - add textbox mirroring existing shell output
# - add peak labels to area peaks
# - add option for setting Gaussian filter std dev