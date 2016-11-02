from matplotlib import pyplot as matplot
from matplotlib import animation as matani

def plotfun(z1,z2,maxy):
    fig = matplot.figure()
    axes = matplot.axes(xlim=(0,len(z1)), ylim = (0,maxy))
    line, = axes.plot([],[], lw=2)
    def init():
        line.set_data([], [])
        return line,
    def animate(j):
        x = z1[1:j]
        y = z2[1:j]
        line.set_data(x,y)
        return line,
    anim = matani.FuncAnimation(fig, animate, init_func=init, frames = len(z1), interval=20, blit=True)
    matplot.show()