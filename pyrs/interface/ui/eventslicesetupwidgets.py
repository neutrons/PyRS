import numpy as np
import bisect
try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QMenu, QAction
    from PyQt5.QtGui import QCursor
except ImportError:
    from PyQt4 import QtCore
    from PyQt4.QtGui import QMenu, QAction, QCursor
    
    
import mplgraphicsview

COLOR_LIST = ['red', 'green', 'black', 'cyan', 'magenta', 'yellow']


class LogGraphicsView(mplgraphicsview.MplGraphicsView):
    """
    Class ... extends ...
    for specific needs of the graphics view for interactive plotting of sample log,

    Note:
    1. each chopper-slicer picker is a vertical indicator
       (ideally as a picker is moving, a 2-way indicator can be shown on the canvas
    """
    # define signals
    mySlicerUpdatedSignal = QtCore.pyqtSignal(list)  # signal as the slicers updated

    def __init__(self, parent):
        """
        Purpose
        :return:
        """
        # Base class constructor
        mplgraphicsview.MplGraphicsView.__init__(self, parent)

        # parent window (logical parent)
        self._myParent = None

        # GUI property
        self.menu = None

        # collection of indicator IDs that are on canvas
        self._currentLogPickerList = list()   # list of indicator IDs.
        self._pickerRangeDict = dict()  # dictionary for picker range. key: position, value: indicator IDs

        # resolution to find
        self._resolutionRatio = 0.001  # resolution to check mouse position
        self._pickerRangeRatio = 0.01  # picker range = (X_max - X_min) * ratio
        self._pickerRange = None  # picker range
        self._currXLimit = (0., 1.)  # 2-tuple as left X limit and right X limit

        # manual slicer picker mode
        self._inManualPickingMode = False
        # mouse mode
        self._mouseLeftButtonHold = False

        # current plot IDs
        self._currPlotID = None
        self._currentSelectedPicker = None
        self._currMousePosX = None
        self._currMousePosY = None

        # about a selected picker
        self._leftPickerLimit = None
        self._rightPickerLimit = None

        # register dictionaries
        self._sizeRegister = dict()

        # extra title message
        self._titleMessage = ''

        # container for segments plot
        self._splitterSegmentsList = list()

        # define the event handling methods
        self._myCanvas.mpl_connect('button_press_event', self.on_mouse_press_event)
        self._myCanvas.mpl_connect('button_release_event', self.on_mouse_release_event)
        self._myCanvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

        return

    def _calculate_distance_to_nearest_indicator(self, pos_x):
        """
        calculate the distance between given position X to its nearest indicator
        :param pos_x:
        :return: 2-tuple.  nearest distance and indicator ID of the nearest indicator
        """
        def nearest_data(array, x):
            """
            find out the nearest value in a sorted array against given X
            :param array:
            :param x:
            :return: distance and the index of the nearest item in the array
            """
            right_index = bisect.bisect_left(array, x)
            left_index = right_index - 1

            if left_index < 0:
                # left to Index=0
                nearest_index = 0
                distance = array[0] - x
            elif right_index == len(array):
                # right to Index=-1
                nearest_index = left_index
                try:
                    distance = x - array[left_index]
                except TypeError as type_err:
                    print '[DB...BAT] x = {0}, array = {1}'.format(x, array)
                    raise type_err
            else:
                dist_left = x - array[left_index]
                dist_right = array[right_index] - x
                if dist_left < dist_right:
                    nearest_index = left_index
                    distance = dist_left
                else:
                    nearest_index = right_index
                    distance = dist_right
            # END-IF-ELSE

            return distance, nearest_index
        # END-DEF

        # return very large number if there is no indicator on canvas
        if len(self._pickerRangeDict) == 0:
            return 1.E22, -1

        # get the indicator positions
        picker_pos_list = self._pickerRangeDict.keys()
        picker_pos_list.sort()

        nearest_picker_distance, nearest_item_index = nearest_data(picker_pos_list, pos_x)
        nearest_picker_position = picker_pos_list[nearest_item_index]
        nearest_picker_id = self._pickerRangeDict[nearest_picker_position]

        return nearest_picker_distance, nearest_picker_id

    def _remove_picker_from_range_dictionary(self, value_to_remove):
        """
        remove an entry in the dictionary by value
        :param value_to_remove:
        :return:
        """
        self._pickerRangeDict = {pos_x: picker_id for pos_x, picker_id in
                                 self._pickerRangeDict.items() if picker_id != value_to_remove}

        return

    def clear_picker(self):
        """
        blabla  will trigger everything including rewrite table!
        :return:
        """
        # remove the picker from the list
        for p_index, picker in enumerate(self._currentLogPickerList):
            # remove from dictionary
            self._remove_picker_from_range_dictionary(picker)
            # remove from canvas
            self.remove_indicator(picker)

        # reset
        self._currentSelectedPicker = None
        self._currentLogPickerList = list()

        # update the new list to parent window
        picker_pos_list = list()
        self.mySlicerUpdatedSignal.emit(picker_pos_list)

        return

    def deselect_picker(self):
        """
        de-select the picker by changing its color and reset the flat
        :return:
        """
        assert self._currentSelectedPicker is not None, 'There is no picker that is selected to de-select.'

        self.update_indicator(self._currentSelectedPicker, 'red')
        self._currentSelectedPicker = None

        return

    def get_data_range(self):
        """ Get data range from the 1D plots on canvas
        :return: 4-tuples as min_x, max_x, min_y, max_y
        """
        if len(self._sizeRegister) == 0:
            raise RuntimeError('Unable to get data range as there is no plot on canvas')

        x_min_list = list()
        x_max_list = list()
        y_min_list = list()
        y_max_list = list()

        for value_tuple in self._sizeRegister.values():
            x_min, x_max, y_min, y_max = value_tuple
            x_min_list.append(x_min)
            x_max_list.append(x_max)
            y_min_list.append(y_min)
            y_max_list.append(y_max)
        # END-FOR

        x_min = min(np.array(x_min_list))
        x_max = max(np.array(x_max_list))
        y_min = min(np.array(y_min_list))
        y_max = max(np.array(y_max_list))

        return x_min, x_max, y_min, y_max

    def get_pickers_positions(self):
        """
        get the positions of all pickers on canvas
        :return: a list of floats
        """
        picker_pos_list = list()
        for p_id in self._currentLogPickerList:
            pos = self.get_indicator_position(p_id)[0]
            picker_pos_list.append(pos)

        picker_pos_list.sort()

        return picker_pos_list

    def menu_add_picker(self):
        """
        add a picker (an indicator) and update the list of pickers' positions to parent
        :return:
        """
        self.add_picker(self._currMousePosX)
        # # add a picker
        # indicator_id = self.add_vertical_indicator(self._currMousePosX, color='red', line_width='2')
        # # add to dictionary
        # self._currentLogPickerList.append(indicator_id)
        # # add the picker to the dictionary
        # self._pickerRangeDict[self._currMousePosX] = indicator_id
        #
        # # update the new list to parent window
        # picker_pos_list = self.get_pickers_positions()
        # self.mySlicerUpdatedSignal.emit(picker_pos_list)

        return

    def add_picker(self, pos_x):
        """
        add a log picker
        :return:
        """
        # add a picker
        indicator_id = self.add_vertical_indicator(pos_x, color='red', line_width='2')
        # add to dictionary
        self._currentLogPickerList.append(indicator_id)
        # add the picker to the dictionary
        # self._pickerRangeDict[self._currMousePosX] = indicator_id
        self._pickerRangeDict[pos_x] = indicator_id

        # update the new list to parent window
        picker_pos_list = self.get_pickers_positions()
        self.mySlicerUpdatedSignal.emit(picker_pos_list)

    def menu_delete_picker(self):
        """
        delete the selected picker
        :return:
        """
        # check
        if self._currentSelectedPicker is None:
            raise RuntimeError('The prerequisite to delete a picker is to have an already-selected picker.')

        # remove the picker from the list
        p_index = self._currentLogPickerList.index(self._currentSelectedPicker)
        self._currentLogPickerList.pop(p_index)
        # remove from dictionary
        self._remove_picker_from_range_dictionary(self._currentSelectedPicker)
        # remove from canvas
        self.remove_indicator(self._currentSelectedPicker)

        # reset
        self._currentSelectedPicker = None

        # update the new list to parent window
        picker_pos_list = self.get_pickers_positions()
        self.mySlicerUpdatedSignal.emit(picker_pos_list)

        return

    def on_mouse_press_event(self, event):
        """
        determine whether the mode is on
        right button:
            pop out menu if it is relevant
        left button:
            get start to
        :param event:
        :return:
        """
        # only respond when in manual picking mode
        if not self._inManualPickingMode:
            return

        button = event.button
        self._currMousePosX = event.xdata

        if button == 1:
            # left button: if a picker is selected then enter on hold mode
            if self._currentSelectedPicker is not None:
                self._mouseLeftButtonHold = True

        elif button == 3:
            # right button: pop-out menu
            self.menu = QMenu(self)

            if self._currentSelectedPicker is None:
                # no picker is selected
                action1 = QAction('Add Picker', self)
                action1.triggered.connect(self.menu_add_picker)
                self.menu.addAction(action1)

            else:
                # some picker is selected
                action2 = QAction('Delete Picker', self)
                action2.triggered.connect(self.menu_delete_picker)
                self.menu.addAction(action2)

            # add other required actions
            self.menu.popup(QCursor.pos())
        # END-IF-ELSE

        return

    def on_mouse_release_event(self, event):
        """
        left button:
            release the hold-picker mode
        :param event:
        :return:
        """
        # do not respond if it is not in manual picker setup mode
        if not self._inManualPickingMode:
            return

        # determine button and position
        button = event.button

        if button == 1:
            # left button: terminate the state for being on hold
            self._mouseLeftButtonHold = False

        # END-IF

        return

    def on_mouse_motion(self, event):
        """
        If left-button is on hold (and thus a picker is selected, them move the picker)
        otherwise, check whether the cursor is close to any picker enough to select it or far away enough to deselect
                previously selected
        :param event:
        :return:
        """
        # return if not in manual mode
        if not self._inManualPickingMode:
            return

        # return if out of boundary
        if event.xdata is None or event.ydata is None:
            return

        # determine button and position
        # button = event.button
        self._currMousePosX = event.xdata
        self._currMousePosY = event.ydata

        # determine the right position and left position with update of
        if self._currXLimit != self.getXLimit():
            self._currXLimit = self.getXLimit()
            delta_x = self._currXLimit[1] - self._currXLimit[0]
            self._pickerRange = delta_x * self._pickerRangeRatio * 0.5

        # check status
        if self._mouseLeftButtonHold:
            # mouse button is hold with a picker is selected
            assert self._currentSelectedPicker is not None, 'In mouse-left-button-hold mode, a picker must be selected.'

            # check whether the selected picker can move
            print '[DB...BAT] Left limit = ', self._leftPickerLimit, ', Range = ', self._pickerRange
            left_bound = self._leftPickerLimit + self._pickerRange
            right_bound = self._rightPickerLimit - self._pickerRange
            if left_bound < self._currMousePosX < right_bound:
                # free to move
                self.set_indicator_position(self._currentSelectedPicker, pos_x=self._currMousePosX,
                                            pos_y=self._currMousePosY)
                # update the position dictionary
                self._remove_picker_from_range_dictionary(self._currentSelectedPicker)
                self._pickerRangeDict[self._currMousePosX] = self._currentSelectedPicker

                # update the pickers' positions with parent window
                picker_pos_list = self.get_pickers_positions()
                self.mySlicerUpdatedSignal.emit(picker_pos_list)

            else:
                # unable to move anymore: quit the hold and select to move mode
                self.deselect_picker()
                self._mouseLeftButtonHold = False
            # END-IF-ELSE
        else:
            # mouse button is not hold so need to find out whether the mouse cursor in in a range
            distance, picker_id = self._calculate_distance_to_nearest_indicator(self._currMousePosX)
            # print '[DB...BAT] distance = {0}, Picker ID = {1}, Picker-range = {2}'.format(distance, picker_id,
            #                                                                               self._pickerRange)
            if distance < self._pickerRange:
                # in the range: select picker
                self.select_picker(picker_id)
            elif self._currentSelectedPicker is not None:
                # in the range: deselect picker
                self.deselect_picker()
            # END-IF-ELSE
        # END-IF-ELSE

        return

    def select_picker(self, picker_id):
        """
        select a slicer picker (indicator) on the canvas
        :param picker_id:
        :return:
        """
        # return if it is the same picker that is already chosen
        if self._currentSelectedPicker == picker_id:
            return

        # previously selected: de-select
        if self._currentSelectedPicker is not None:
            self.deselect_picker()

        # select current on
        self._currentSelectedPicker = picker_id
        self.update_indicator(self._currentSelectedPicker, color='blue')

        # define the pickers to its left and right for boundary
        curr_picker_pos = self.get_indicator_position(picker_id)[0]
        picker_pos_list = sorted(self._pickerRangeDict.keys())
        pos_index = picker_pos_list.index(curr_picker_pos)

        # get the data range for the left most or right most boundary
        x_min, x_max, y_min, y_max = self.get_data_range()

        # determine left boundary
        if pos_index == 0:
            # left most indicator. set the boundary to data's min X
            self._leftPickerLimit = x_min - self._pickerRange
        else:
            self._leftPickerLimit = picker_pos_list[pos_index-1]

        # determine the right boundary
        if pos_index == len(picker_pos_list) - 1:
            # right most indicator. set the boundary to data's max X
            self._rightPickerLimit = x_max + self._pickerRange
        else:
            self._rightPickerLimit = picker_pos_list[pos_index+1]

        return

    def plot_sample_log(self, vec_x, vec_y, sample_log_name):
        """ Purpose: plot sample log

        Guarantee: canvas is replot
        :param vec_x
        :param vec_y
        :param sample_log_name:
        :return:
        """
        # check
        assert isinstance(vec_x, np.ndarray), 'VecX must be a numpy array but not %s.' \
                                              '' % vec_x.__class__.__name__
        assert isinstance(vec_y, np.ndarray), 'VecY must be a numpy array but not %s.' \
                                              '' % vec_y.__class__.__name__
        assert isinstance(sample_log_name, str)

        # set label
        try:
            the_label = '%s Y (%f, %f)' % (sample_log_name, min(vec_y), max(vec_y))
        except TypeError as type_err:
            err_msg = 'Unable to generate log with %s and %s: %s' % (
                str(min(vec_y)), str(max(vec_y)), str(type_err))
            raise TypeError(err_msg)

        # add plot and register
        plot_id = self.add_plot_1d(vec_x, vec_y, label='', marker='.', color='blue', show_legend=False)
        self.set_title(title=the_label)
        self._sizeRegister[plot_id] = (min(vec_x), max(vec_x), min(vec_y), max(vec_y))

        # auto resize
        self.resize_canvas(margin=0.05)

        # update
        self._currPlotID = plot_id

        return

    def remove_slicers(self):
        """
        remove slicers
        :return:
        """
        for slicer_plot_id in self._splitterSegmentsList:
            self.remove_line(slicer_plot_id)

        # clear
        self._splitterSegmentsList = list()

        return

    def reset(self):
        """
        Reset canvas
        :return:
        """
        # dictionary
        self._sizeRegister.clear()

        # clear slicers
        self.remove_slicers()

        # clear all lines
        self.clear_all_lines()
        self._currPlotID = None

        return

    def resize_canvas(self, margin):
        """

        :param margin:
        :return:
        """
        # get min or max
        try:
            x_min, x_max, y_min, y_max = self.get_data_range()
        except RuntimeError:
            # no data left on canvas
            canvas_x_min = 0
            canvas_x_max = 1
            canvas_y_min = 0
            canvas_y_max = 1
        else:
            # get data range
            range_x = x_max - x_min
            canvas_x_min = x_min - 0.05 * range_x
            canvas_x_max = x_max + 0.05 * range_x

            range_y = y_max - y_min
            canvas_y_min = y_min - 0.05 * range_y
            canvas_y_max = y_max + 0.05 * range_y
        # END-IF-ELSE()

        # resize canvas
        self.setXYLimit(xmin=canvas_x_min, xmax=canvas_x_max, ymin=canvas_y_min, ymax=canvas_y_max)

        return

    def set_parent_window(self, parent_window):
        """
        set the parent window (logically parent but not widget in the UI)
        :param parent_window:
        :return:
        """
        self._myParent = parent_window

        # connect signal
        self.mySlicerUpdatedSignal.connect(self._myParent.evt_rewrite_manual_table)

        return

    def set_manual_slicer_setup_mode(self, mode_on):
        """
        set the canvas in the mode to set up slicer manually
        :param mode_on:
        :return:
        """
        assert isinstance(mode_on, bool), 'Mode on/off {0} must be a boolean but not a {1}.' \
                                          ''.format(mode_on, type(mode_on))
        self._inManualPickingMode = mode_on

        # reset all the current-on-select variables
        if mode_on:
            # TODO/ISSUE/33 - Add 2 pickers/indicators at Time[0] and Time[-1] if the table is empty
            pass

            # TODO/ISSUE/33 - Add pickers to if pickers are hidden!
            pass

        else:
            # TODO/ISSUE/33 - Hide all the pickers
            if self._currentSelectedPicker is not None:
                # de-select picker
                self.deselect_pikcer(self._currentSelectedPicker)
                self._currentSelectedPicker = None
            self._currMousePosX = None
            self._currMousePosY = None

        return

    def show_slicers(self, vec_times, vec_target_ws):
        """
        show slicers on the canvas
        :param vec_times:
        :param vec_target_ws:
        :return:
        """
        # check state
        if self._currPlotID is None:
            return True, 'No plot on the screen yet.'

        assert len(vec_times) == len(vec_target_ws) + 1, 'Assumption that input is a histogram!'

        # get data from the figure
        vec_x, vec_y = self.canvas().get_data(self._currPlotID)

        num_color = len(COLOR_LIST)

        # if there are too many slicing segments, then only shows the first N segments
        MAX_SEGMENT_TO_SHOW = 20
        num_seg_to_show = min(len(vec_target_ws), MAX_SEGMENT_TO_SHOW)

        for i_seg in range(num_seg_to_show):
            # get start time and stop time
            x_start = vec_times[i_seg]
            x_stop = vec_times[i_seg+1]
            color_index = vec_target_ws[i_seg]

            # get start time and stop time's index
            i_start = (np.abs(vec_x - x_start)).argmin()
            i_stop = (np.abs(vec_x - x_stop)).argmin()
            if i_start == i_stop:
                # empty!
                print '[DB...WARNING] Range: %d to %d  (%f to %f) cannot generate any vector. ' \
                      '' % (i_start, i_stop, vec_x[i_start], vec_x[i_stop])
                continue
            elif i_start > i_stop:
                raise RuntimeError('It is impossible to have start index {0} > stop index {1}'
                                   ''.format(i_start, i_stop))

            # get the partial for plot
            vec_x_i = vec_x[i_start:i_stop]
            vec_y_i = vec_y[i_start:i_stop]

            # plot
            color_i = COLOR_LIST[color_index % num_color]
            seg_plot_index = self.add_plot_1d(vec_x_i, vec_y_i, marker=None, line_style='-', color=color_i,
                                              line_width=2)

            self._splitterSegmentsList.append(seg_plot_index)

        # END-FOR

        status = True
        error_msg = None
        if len(vec_target_ws) > MAX_SEGMENT_TO_SHOW:
            status = False
            error_msg = 'There are too many (%d) segments in the slicers.  Only show the first %d.' \
                        '' % (len(vec_target_ws), MAX_SEGMENT_TO_SHOW)

        return status, error_msg
