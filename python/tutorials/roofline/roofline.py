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

L1_PEAK = 58776.
L2_PEAK = 7817.
HBM_PEAK = 1935.
PERFORMANCE_PEAK = 234.

fig = plt.figure(1, figsize=(10.67, 6.6))
plt.clf()
patch_handles = list()

def roofline(idx, tag, filename, FLOPS, AIHBM, AIL2=None, AIL1=None, LABELS=None, flag='all'):

    if not FLOPS:
        print('FLOPS can not be empty!')
        return
    if max(FLOPS) == 0:
        print('FLOPS are all 0s!')
        return
    if (not AIHBM) and (not AIL2) and (not AIL1):
        print('AIHBM, AIL2 and AIL1 can not all be empty!')
        return
    if (len(FLOPS) != len(AIHBM)) or (len(FLOPS) != len(AIL2)) or (len(FLOPS) != len(AIL1)):
        print('FLOPS needs to have the same length as AI!')
        return
    if (flag != 'HBM') and (flag != 'L2') and (flag != 'L1') and (flag != 'all'):
        print('flag needs to be one of HBM, L2, L1, and all!')
        return
    LABELS = [LABELS]

    memRoofs = [('L1', L1_PEAK), ('L2', L2_PEAK),  ('HBM', HBM_PEAK)]
    cmpRoofs = [('Tensor', PERFORMANCE_PEAK)]

    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
    ax.set_ylabel('Performance [GFLOP/sec]')

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)

    ixx = int(nx*0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scomp_x_elbow = []
    scomp_ix_elbow = []
    smem_x_elbow = []
    smem_ix_elbow = []

    x = np.logspace(xmin, xmax, nx)
    for roof in cmpRoofs:
        for ix in range(1, nx):
            if float(memRoofs[0][1] * x[ix]) >= roof[1]*1024 and (memRoofs[0][1] * x[ix-1]) < roof[1]*1024:
                scomp_x_elbow.append(x[ix-1])
                scomp_ix_elbow.append(ix-1)
                break

    for roof in memRoofs:
        for ix in range(1, nx):
            if (cmpRoofs[0][1]*1024 <= roof[1] * x[ix] and cmpRoofs[0][1]*1024 > roof[1] * x[ix-1]):
                smem_x_elbow.append(x[ix-1])
                smem_ix_elbow.append(ix-1)
                break

    for i in range(len(cmpRoofs)):
        roof = cmpRoofs[i][1]*1024
        y = np.ones(len(x)) * roof
        ax.plot(x[scomp_ix_elbow[i]:],
                y[scomp_ix_elbow[i]:], c='k', ls='-', lw='2')

    for i in range(len(memRoofs)):
        roof = memRoofs[i][1]
        y = x * roof
        ax.plot(x[:smem_ix_elbow[i]+1],
                y[:smem_ix_elbow[i]+1], c='k', ls='-', lw='2')
        
    # a lambda function to calculate roofline boundary
    # x: arithmetic intensity
    def roofline_boundary(x, peak):
        return min(1000 * PERFORMANCE_PEAK, peak * x)

    for i in range(len(AIHBM)):
        if flag == 'all':
            # plot a line with flops
            ax.axhline(y=float(FLOPS[i]), c=colors[idx % 10], ls='--', lw='1')
            ax.text(xlim[-1]*1.1, float(FLOPS[i]), f'{FLOPS[i]/1000:.1f} TFLOP/s', fontsize=8, color=colors[idx % 10])
             
            ax.plot(float(AIL1[i]), float(FLOPS[i]), c=colors[idx % 10], marker=styles[0],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=tag)

            ax.plot(float(AIL2[i]), float(FLOPS[i]), c=colors[idx % 10], marker=styles[1],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=tag)

            ax.plot(float(AIHBM[i]), float(FLOPS[i]), c=colors[idx % 10], marker=styles[2],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=tag)

            ax.text(xlim[0]*1.1, float(FLOPS[i]), f"L1 AI:{AIL1[i]:.1f}, {FLOPS[i] / roofline_boundary(AIL1[i], L1_PEAK) * 100:.2f}%; L2 AI:{AIL2[i]:.1f}, {FLOPS[i] / roofline_boundary(AIL2[i], L2_PEAK) * 100:.2f}%; HBM AI:{AIHBM[i]:.1f}, {FLOPS[i] / roofline_boundary(AIHBM[i], HBM_PEAK) * 100:.2f}%", fontsize=8, color=colors[idx % 10])
        else:
            print(f"flag must be 'all'!")
            return

    marker_handles = []
    if flag == 'all':
        for i in range(len(memRoofs)):
            marker_handles.append(ax.plot([], [], c='k', marker=styles[i], linestyle='None', ms=markersize,
                                  markerfacecolor='none', markeredgewidth=markerwidth, label=memRoofs[i][0])[0])      
    else:
        print(f"flag must be 'all'!")
        return

    for roof in cmpRoofs:
        ax.text(x[-ixx], roof[1]*1024,
                roof[0] + ': ' + '{0:.1f}'.format(roof[1]) + ' TFLOP/s',
                horizontalalignment='right',
                verticalalignment='bottom')

    for roof in memRoofs:
        ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0])
                        * fig.get_size_inches()[1]/fig.get_size_inches()[0])
        if x[ixx]*roof[1] > ymin:
            ax.text(x[ixx], x[ixx]*roof[1]*(1+0.25*np.sin(ang)**2),
                    roof[0] + ': ' +
                    '{0:.1f}'.format(float(roof[1])) + ' GB/s',
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
                    '{0:.1f}'.format(float(roof[1])) + ' GB/s',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    rotation=180/np.pi*ang)

    leg1 = plt.legend(handles=marker_handles, loc='lower right', ncol=len(
        flag[0]) if 'all' not in flag else 3, bbox_to_anchor=(1, 0))
    ax.add_artist(leg1)

    patch_handles.append(mpatches.Patch(
                color=colors[idx % 10], label=tag))

    leg2 = plt.legend(handles=patch_handles, loc=4, ncol=1,
                      bbox_to_anchor=(1, 0.1), scatterpoints=1)
    ax.add_artist(leg2)
    
    ax.text(xlim[0]*1.1, ylim[1]/1.1, '-'.join([filename, flag]),
            horizontalalignment='left', verticalalignment='top')

    plt.savefig('picture/' + '_'.join([filename, flag])+'.png')
