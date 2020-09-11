import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np
from skimage import filters, exposure
import scipy.ndimage as ndimage
from frozendict import frozendict
from matplotlib.path import Path

class ZStackViewer():
    def __init__(self, img, width=6, height=6):
        self.width = width
        self.height = height
        self.img = img
        self.index = 0
        self._remove_keymap_conflicts({'j', 'k'})
        self.artists = []
        self.lines = []
        self.points = {}
        pass

    def view_stack(self):
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.imshow(self.img[self.index])
        ax.set_title("Z slice %d" % self.index)
        fig.canvas.mpl_connect('key_press_event', self._process_key)
        plt.show()

    def view_max_projection(self, axis = 0):
        _, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_title("Max Projection")
        max_projection = self.img.max(axis=axis)
        ax.imshow(max_projection)
        plt.show()

    def view_thresholded_max_projection(self, axis = 0):
        _, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_title("Max Projection")
        max_projection = self.img.max(axis=axis)
        max_threshed = max_projection > filters.threshold_otsu(max_projection)
        ax.imshow(max_threshed)
        plt.show()

    def mark_ap_axis(self, ap_coord=2, percentiles=(2,98)):
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        max_projection_z = self.img.max(axis=0)
        max_projection_ap = self.img.max(axis=ap_coord)
        p_low, p_high = np.percentile(max_projection_z, percentiles)
        ax.imshow(exposure.rescale_intensity(max_projection_z, in_range=(p_low, p_high)))
        ax.set_title("Max Projection Z")
        xy_points = plt.ginput(2)
        xs, ys = zip(*xy_points)
        xs = np.round(xs).astype(int)
        ys = np.round(ys).astype(int)
        ax.plot(xs, ys, 'r+')
        fig.canvas.draw()

        

        fig2, ax1 = plt.subplots(figsize=(self.width, self.height))
        p2, p98 = np.percentile(max_projection_ap, (2,98))
        ax1.imshow(exposure.rescale_intensity(max_projection_ap, in_range=(p2, p98)))
        if ap_coord == 2:
            ax1.axvline(ys[0], color='red')
            ax1.axvline(ys[1], color='red')
            ax1.set_title("Max Projection X")
        else:
            ax1.axvline(xs[0], color='red')
            ax1.axvline(xs[1], color='red')
            ax1.set_title("Max Projection Y")            
        yz_points = plt.ginput(2)
        ys2, zs = zip(*yz_points)
        zs = np.round(zs).astype(int)
        ax1.plot(ys2, zs, 'r+')
        fig2.canvas.draw()

        self.ap_axes_ends = np.array([[zs[0], ys[0], xs[0]],
                                    [zs[1], ys[1], xs[1]]])


    
    def select_points(self):
        self.curr_point_artist = None
        self.points = {}
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        p2, p98 = np.percentile(self.img[self.index], (2,98))
        ax.imshow(exposure.rescale_intensity(self.img[self.index], in_range=(p2, p98)))
        ax.set_title("Z slice %d" % self.index)     
        fig.canvas.mpl_connect('button_press_event', self._mark_and_record_points)
        fig.canvas.mpl_connect('key_press_event', self._process_key_points)
        plt.show()
        points = []
        for z_slice in self.points.keys():
            for pt in self.points[z_slice]:
                points.append([z_slice, pt[1], pt[0]])
        return np.array(points).T

    def select_points_max_proj(self):
        self.curr_point_artist = None
        self.points = []
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        z_locations = self.img.argmax(axis=0)
        max_proj = self.img.max(axis=0)
        p2, p98 = np.percentile(max_proj, (2, 98))
        ax.imshow(exposure.rescale_intensity(max_proj, in_range=(p2, p98)))
        fig.canvas.mpl_connect('button_press_event', self._mark_and_record_points_maxproj)
        plt.show()
        points = []
        for point in self.points:
            x = int(np.round(point[0]))
            y = int(np.round(point[0]))
            points.append([z_locations[y, x], y, x])
        return np.array(points).T     
    
    def _mark_and_record_points_maxproj(self, event):
        self.points.append((event.xdata, event.ydata))
        fig = event.canvas.figure
        ax = fig.axes[0]
        xs, ys = zip(*self.points)
        if self.curr_point_artist is not None:
            self.artists[-1].remove()
            self.artists = self.artists[:-1]
        self.curr_point_artist = ax.plot(xs, ys, 'rx')[0]
        self.artists.append(self.curr_point_artist)
        fig.canvas.draw()

    def _mark_and_record_points(self, event):
        if self.index not in self.points.keys():
            self.points[self.index] = []
        self.points[self.index].append((event.xdata, event.ydata))
        fig = event.canvas.figure
        ax = fig.axes[0]
        xs, ys = zip(*self.points[self.index])
        if self.curr_point_artist is not None:
            self.artists[-1].remove()
            self.artists = self.artists[:-1]
        self.curr_point_artist = ax.plot(xs, ys, 'rx')[0]
        self.artists.append(self.curr_point_artist)
        fig.canvas.draw()

    
    def _start_line_draw(self, event):
        self.line_x = [event.xdata, event.xdata]
        self.line_y = [event.ydata, event.ydata]
        fig = event.canvas.figure
        ax = fig.axes[0]
        self.curr_line = ax.plot(self.line_x, self.line_y, color="red")[0]
        self.artists.append(self.curr_line)
        fig.canvas.draw()

    def _draw_line(self, event):
        if self.line_x is not None:
            self.line_x[1] = event.xdata
            self.line_y[1] = event.ydata
            self.curr_line.set_xdata(self.line_x)
            self.curr_line.set_ydata(self.line_y)
            self.curr_line.figure.canvas.draw()
    
    def _end_line_draw(self, event):
        data = [self.line_x[0], self.line_x[1], self.line_y[0], self.line_y[1], self.index]
        self.lines.append(data)
        self.line_x = None
        self.line_y = None


    def _previous_slice(self, ax):
        self.index = (self.index - 1) % self.img.shape[0]
        ax.set_title("Z slice %d" % self.index)
        ax.images[0].set_array(self.img[self.index])

    def _next_slice(self, ax):
        self.index = (self.index + 1) % self.img.shape[0]
        ax.set_title("Z slice %d" % self.index)
        ax.images[0].set_array(self.img[self.index])

    def _process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self._previous_slice(ax)
            for artist in self.artists:
                artist.remove()
            self.artists = []
        elif event.key == 'k':
            self._next_slice(ax)
            for artist in self.artists:
                artist.remove()
            self.artists = []
        elif event.key == 'd':
            if len(self.lines) > 0:
                self.curr_line.remove()
                self.artists = self.artists[:-1]
                self.lines = self.lines[:-1]
        fig.canvas.draw()

    def _process_key_points(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self._previous_slice(ax)
            for artist in self.artists:
                artist.remove()
            self.artists = []
            self.curr_point_artist = None
        elif event.key == 'k':
            self._next_slice(ax)
            for artist in self.artists:
                artist.remove()
            self.artists = []
            self.curr_point_artist = None
        elif event.key == 'd':
            if len(self.lines) > 0:
                self.curr_line.remove()
                self.artists = self.artists[:-1]
                self.points[self.index] = self.points[self.index][:-1]
        fig.canvas.draw()

    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

class HyperStackViewer(ZStackViewer):
    T = 0
    Z = 1
    C = 2

    def __init__(self, img, width=6, height=6):
        super().__init__(img, width, height)
        self._remove_keymap_conflicts({'j', 'k', 'd', 'z', 'x', 'c', 'u', 'i'})
        # Format is T Z C X Y
        self.index = [0, 0, 0]
        self.rgb_on = False
        pass

    def select_region_clicky(self):
        self.curr_point_artist = None
        self.points = []
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.imshow(self._get_curr_slice())
        ax.set_title(self._generate_title_string())
        self.fig = fig
        self.cidclick = fig.canvas.mpl_connect('button_press_event', self._mark_and_record_points_clicky)
        self.cidkey = fig.canvas.mpl_connect('key_press_event', self._process_key_points)
        plt.show()
        xs, ys =  zip(*self.points)
        points = [xs, ys]
        return self._points_to_mask(np.array(points).T)
    
    def _points_to_mask(self, points):
        p = Path(points, closed=True)
        x, y = np.meshgrid(np.arange(self.img.shape[3], dtype=int), np.arange(self.img.shape[4], dtype=int))
        pix = np.vstack((x.flatten(), y.flatten())).T
        in_contour = p.contains_points(pix).reshape((self.img.shape[3], self.img.shape[4]))
        return np.tile(in_contour, (self.img.shape[0], self.img.shape[1], self.img.shape[2], 1, 1))

    def _get_curr_slice(self):
        if self.rgb_on:
            if self.img.shape[2] == 3:
                rgb_img = self.img[self.index[0], self.index[1], :, :, :]
            else:
                rgb_img = self.img[self.index[0], self.index[1], :, :, :]
                rgb_img = np.concatenate((np.zeros((3-rgb_img.shape[0], rgb_img.shape[1], rgb_img.shape[2])), rgb_img), axis=0)
            return np.flip(np.moveaxis(rgb_img, 0, 2), axis=2).astype(self.img.dtype)
        else:
            return self.img[self.index[0], self.index[1], self.index[2], :, :]

    def _generate_title_string(self):
        rgb_state = "RGB" if self.rgb_on else str(self.index[2])
        return ("T: %d, Z: %d, C: %s" % (self.index[0], self.index[1], rgb_state))

    def _jog_slice(self, fig, dim, step_size):
        ax = fig.axes[0]
        self.index[dim] = (self.index[dim] + step_size) % self.img.shape[dim]
        ax.set_title(self._generate_title_string())
        sl = self._get_curr_slice()
    
        print(np.max(sl))
        print(np.min(sl))
        ax.images[0].set_array(sl)
        fig.canvas.draw()

    def _process_key_points(self, event):
        fig = event.canvas.figure
        if event.key == 'd':
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self._draw_clicky_contour(fig)
        elif event.key == 'x':
            self.rgb_on = ~self.rgb_on
            self._jog_slice(fig, self.Z, 0)
        else:
            if event.key == 'j':
                self._jog_slice(fig, self.Z, -1)
            elif event.key == 'k':
                self._jog_slice(fig, self.Z, 1)
            elif event.key == 'z':
                self._jog_slice(fig, self.C, -1)
            elif event.key == 'c':
                self._jog_slice(fig, self.C, 1)
            elif event.key == 'u':
                self._jog_slice(fig, self.T, -1)
            elif event.key == 'i':
                self._jog_slice(fig, self.T, 1)

    def _draw_clicky_contour(self, fig):
        ax = fig.axes[0]
        xs, ys = zip(*self.points)
        if self.curr_point_artist is not None:
            self.artists[-1].remove()
            self.artists = self.artists[:-1]
        self.curr_point_artist = ax.plot(xs, ys, 'w-')[0]
        self.artists.append(self.curr_point_artist)
        fig.canvas.draw()

    def _mark_and_record_points_clicky(self, event):
        self.points.append([event.xdata, event.ydata])
        print((event.xdata, event.ydata))
        fig = event.canvas.figure
        self._draw_clicky_contour(fig)
        e = 0.01
        if len(self.points) > 3 and self._distance(self.points[-1], self.points[-2]) < e:
            print("Region closed")
            self._disconnect()
            plt.close()

    def _distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidclick)
        self.fig.canvas.mpl_disconnect(self.cidkey)