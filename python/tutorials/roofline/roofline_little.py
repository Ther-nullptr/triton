import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# hyper parameters

font = {'size': 15}
plt.rc('font', **font)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
styles = ['o', 's', 'v', '^', 'D', ">", "<", "*", "h", "H",
          "+", "1", "2", "3", "4", "8", "p", "d", "|", "_", ".", ","]

markersize = 5
markerwidth = 2
maxchar = 25

nx = 10000
xmin = -3
xmax = 4
ymin = 1
ymax = 1000000

Latency =  1. # ns
PeakIPC = 2.

fig = plt.figure(1, figsize=(10.67 * 2, 6.6 * 2))
plt.clf()
marker_handles = list()
patch_handles = list()

def roofline(idx, tag, filename, ActiveInsts, PracticalIPC):

    latencyRoofs = [('Latency', Latency)]
    ipcRoofs = [('Tensor', PeakIPC)]

    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Active Instructions of type x')
    ax.set_ylabel('IPC of type x')

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)

    ixx = int(nx*0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ipc_x_elbow = []
    ipc_ix_elbow = []
    inst_x_elbow = []
    inst_ix_elbow = []

    # plot the base roofline
    x = np.logspace(xmin, xmax, nx)
    if idx == 0:
        for i, roof in enumerate(ipcRoofs):
            for ix in range(1, nx):
                if float(latencyRoofs[0][1] * x[ix]) >= roof[1]*1024 and (latencyRoofs[0][1] * x[ix-1]) < roof[1]*1024:
                    ipc_x_elbow.append(x[ix-1])
                    ipc_ix_elbow.append(ix-1)
                    break

        for i, roof in enumerate(latencyRoofs):
            for ix in range(1, nx):
                if (ipcRoofs[0][1]*1024 <= roof[1] * x[ix] and ipcRoofs[0][1]*1024 > roof[1] * x[ix-1]):
                    inst_x_elbow.append(x[ix-1])
                    inst_ix_elbow.append(ix-1)
                    break

        for i in range(len(ipcRoofs)):
            roof = ipcRoofs[i][1]*1024
            y = np.ones(len(x)) * roof
            ax.plot(x[ipc_ix_elbow[i]:],
                    y[ipc_ix_elbow[i]:], c='k', ls='-', lw='2')

        for i in range(len(latencyRoofs)):
            roof = latencyRoofs[i][1]
            y = x * roof
            ax.plot(x[:inst_ix_elbow[i]+1],
                    y[:inst_ix_elbow[i]+1], c='k', ls='-', lw='2')
            marker_handles.append(ax.plot([], [], c='k', marker=styles[i], linestyle='None', ms=markersize,
                                    markerfacecolor='none', markeredgewidth=markerwidth, label=latencyRoofs[i][0])[0])

        for roof in ipcRoofs:
            ax.text(x[-ixx], roof[1]*1024,
                    roof[0] + ': ' + '{0:.1f}'.format(roof[1]) + ' (IPC)',
                    horizontalalignment='right',
                    verticalalignment='bottom')

        for roof in latencyRoofs:
            ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0])
                            * fig.get_size_inches()[1]/fig.get_size_inches()[0])
            if x[ixx]*roof[1] > ymin:
                ax.text(x[ixx], x[ixx]*roof[1]*(1+0.25*np.sin(ang)**2),
                        roof[0] + ': ' +
                        '{0:.1f}'.format(float(roof[1])) + ' ns',
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        rotation=180/np.pi*ang)
            else:
                ymin_ix_elbow = list()
                ymin_x_elbow = list()
                for ix in range(1, nx):
                    if (ymin <= roof[1] * x[ix] and ymin > roof[1] * x[ix-1]):
                        ymin_x_elbow.append(x[ix-1])
                        ymin_ix_elbow.append(ix-1)
                        break
                ax.text(x[ixx+ymin_ix_elbow[0]], x[ixx+ymin_ix_elbow[0]]*roof[1]*(1+0.25*np.sin(ang)**2),
                        roof[0] + ': ' +
                        '{0:.1f}'.format(float(roof[1])) + ' ns',
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        rotation=180/np.pi*ang)
        
    # a lambda function to calculate roofline boundary
    # x: arithmetic intensity
    roofline_boundary = lambda x, peak : min(1000 * PeakIPC, peak * x)

    for i in range(len(ActiveInsts)):
        # plot a line with ActiveInsts
        ax.axhline(y=float(ActiveInsts[i]), c=colors[idx % 10], ls='--', lw='1')
        ax.text(xlim[-1]*1.1, float(ActiveInsts[i]), f'{ActiveInsts[i]:.1f}', fontsize=8, color=colors[idx % 10])

        ax.plot(float(ActiveInsts[i]), float(ActiveInsts[i]), c=colors[idx % 10], marker=styles[2],
                linestyle='None', ms=markersize, markerfacecolor='none',
                markeredgewidth=markerwidth, label=tag)

        ax.text(xlim[0]*1.1, float(PracticalIPC[i]), f"Active Instructions: {ActiveInsts[i]:.1f}, {PracticalIPC[i] / roofline_boundary(ActiveInsts[i], Latency) * 100:.2f}%", fontsize=10, color=colors[idx % 10])


    leg1 = plt.legend(handles=marker_handles, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
    ax.add_artist(leg1)

    patch_handles.append(mpatches.Patch(
                color=colors[idx % 10], label=tag))

    leg2 = plt.legend(handles=patch_handles, loc=4, ncol=1, bbox_to_anchor=(1, 0.1), scatterpoints=1)
    ax.add_artist(leg2)
    
    ax.text(xlim[0]*1.1, ylim[1]/1.1, filename, horizontalalignment='left', verticalalignment='top')

    plt.savefig('picture/' + filename + '.png')
