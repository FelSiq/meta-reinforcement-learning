import typing as t

import matplotlib
import matplotlib.pyplot as plt


class MPLRender:
    def __init__(
        self,
        display_delay: float,
        environment_name: t.Optional[str] = None,
        *args,
        **kawrgs
    ):
        self.map = None

        self.plot_environment_name = environment_name

        self.plot_ax1 = None
        self.plot_ax2 = None
        self.plot_fig = None
        self.plot_bg = None
        self.plot_draw_bg = False
        self.plot_display_delay = display_delay

        self.state_values_material = None
        self.state_values = None

    def mpl_render(self):
        if self.plot_fig is None:
            self.plot_fig = plt.figure()
            self.plot_fig.suptitle(self.plot_environment_name)
            self.plot_draw_bg = False

        if self.plot_ax1 is None:
            self.plot_ax1 = self.plot_fig.add_subplot(
                111 if self.state_values is None else 121
            )

        if self.state_values is not None and self.plot_ax2 is None:
            self.plot_ax2 = self.plot_fig.add_subplot(122)
            self.plot_ax1.set_title("Environment view")
            self.plot_ax2.set_title("Estimated state values")

        if not self.plot_draw_bg:
            self.plot_bg = self.plot_ax1.imshow(self.map)
            self.plot_draw_bg = True

        points = self.plot_ax1.plot(
            self.current_pos.x, self.current_pos.y, "o", color="red"
        )

        if self.plot_ax2 is not None:
            self.plot_ax2.imshow(
                self.state_values, cmap="hot", norm=matplotlib.colors.Normalize(0, 1)
            )

        plt.pause(self.plot_display_delay)
        points.pop().remove()

    def mpl_reset(self):
        self.plot_draw_bg = False

    def mpl_close(self):
        if self.plot_fig is not None:
            plt.close(self.plot_fig)

        self.plot_fig = self.plot_ax1 = self.plot_ax2 = None

    def mpl_get_plot_dims(self):
        return self.map.shape
