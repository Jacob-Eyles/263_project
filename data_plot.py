################################################
# data_plot.py

# This file will plot the given data on:
#    - extraction rates (ML/day) - yearly
#    - pressure (MPa)            - 2 yearly
#    - copper conc. (mg/L)       - 5 yearly

# data is given from 1980 to 2018
###############################################
# imports
import numpy as np
from matplotlib import pyplot as plt

#ac_cu_v is the amount of dissolved Cu, in mg/L
ac_cu_y, ac_cu_v = np.loadtxt("ac_cu.csv",\
    delimiter = ",", 
    skiprows = 1,
    unpack = True)

#ac_p_v is the pressure in the aquifer, in MPa
ac_p_y, ac_p_v = np.loadtxt("ac_p.csv",\
    delimiter = ",",
    skiprows = 1,
    unpack = True)

#ac_q_v is the rate of water extraction from the aquifer, in ML/day
ac_q_y, ac_q_v = np.loadtxt("ac_q.csv",\
    delimiter = ",",
    skiprows = 1,
    unpack = True)

x_limits = (1978, 2020)

#prepare the set of plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (19, 6))

#plot extraction rates and pressure over time
p1, = ax1.plot(ac_q_y, ac_q_v)
ax1.set(title = "Water extraction rate and Aquifer pressure\nfor Onehunga aquifer (1980-2018)",
    xlabel = "Year",
    ylabel = "Water extraction rate\n($10^6$ L/day)",
    xlim = x_limits)
ax1.yaxis.label.set_color(p1.get_color())
ax1.tick_params(axis = "y", colors = p1.get_color())

twin1 = ax1.twinx()
p2, = twin1.plot(ac_p_y, ac_p_v, color = "black")
twin1.set(ylabel = "Pressure (MPa)")
twin1.yaxis.label.set_color(p2.get_color())
twin1.tick_params(axis = "y", colors = p2.get_color())

ax1.legend([p1, p2], ["Water extraction rate", "Pressure"],\
    loc = "center right")

#plot pressure and copper concentration over time
p3, = ax2.plot(ac_cu_y, ac_cu_v, color = "#b87333")
ax2.set(title = "Aquifer pressure and copper conc.\nfor Onehunga aquifer (1980-2018)",
    xlabel = "Year",
    ylabel = "Copper conc.(mg/L)",
    xlim = x_limits)
ax2.yaxis.label.set_color("#b87333")
ax2.tick_params(axis = "y", colors = "#b87333")

twin2 = ax2.twinx()
p4, = twin2.plot(ac_p_y, ac_p_v, color = 'k')
twin2.set(ylabel = "Pressure (MPa)")
twin2.yaxis.label.set_color(p4.get_color())
twin2.tick_params(axis = "y", colors = 'k')

ax2.legend([p3, p4], ["Copper concentration","Pressure"],\
    loc = "center right")

# save and show
# plt.show()
plt.savefig("Plot of Aquifer data")