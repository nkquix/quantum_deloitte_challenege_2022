from scipy.io import netcdf as ncf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import os
import gc
import sys
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# Load all the data 
ncf_file = ncf.netcdf_file("aCCF_0623_p_spec.nc") # Read .nc file 
ncf_file
ncf_dict = {key:ncf_file.variables[key][:] for key in ncf_file.variables.keys()} # Change the nc file to dict of numpy nd arrays
flight_data = pd.read_csv("flights.csv",sep=';') # load the flight data
flight_data
bada_data_cruise = pd.read_csv("bada_data_cruise.csv",sep=';') # load cruising data
bada_data_cruise = bada_data_cruise.set_index('FL')
bada_data_climb = pd.read_csv("bada_data_climb.csv",sep=';') # load climbing data
bada_data_climb = bada_data_climb.set_index('FL')
bada_data_descent = pd.read_csv("bada_data_descent.csv",sep=';') # load descent data
bada_data_descent = bada_data_descent.set_index('FL')
# knots to feet/min =>  1 knots = 101.269 feet/min or 0.0308667 Km/min
bada_data_cruise['tas km/min'] = bada_data_cruise["TAS [kts]"]*0.0308
bada_data_descent["ROD [km/min]"] = bada_data_descent["ROD [ft/min]"]*0.0003
bada_data_climb["ROC [km/min]"] = bada_data_climb["ROC [ft/min]"]*0.0003
# knots to feet/min =>  1 knots = 101.269 feet/min or 0.0308667 Km/min

g = 32 # gravitational acceleration in ft/s^2
R = 6373.0  # Radius of eath in km.



# Get cruise level speed
def cruise_lvl_speed(x,dfs_cruise = bada_data_cruise):
	count = 0
	while count>=0:
		try:
			y = dfs_cruise.loc[x,'tas km/min']
			return y
		except Exception as e:
			x += ((-1)**(count))*((count+1)*5)
			count += 1
			if count > 20:
				break
	return np.nan

# Get non cruise level speed
def non_cruise_lvl_speed(x,z_sign,dfs_climb = bada_data_climb,dfs_descent = bada_data_descent):
	count = 0
	if z_sign == 1 :
		dfs =  dfs_climb
		col_name = 'ROC [km/min]'
	else:
		dfs =  dfs_descent
		col_name = 'ROD [km/min]'
	while count>=0:
		try:
			y = dfs.loc[x,col_name]
			return y
		except Exception as e:
			x += ((-1)**(count))*((count+1)*5)
			count += 1
			if count > 20:
				break
	return np.nan

# Get cruise level fuel consumption
def cruise_lvl_fuel(x,dfs_cruise = bada_data_cruise):
	count = 0
	while count>=0:
		try:
			try:
				y = float(dfs_cruise.loc[x,'fuel [kg/min]'])
			except Exception as e:
				y = dfs_cruise.loc[x,'fuel [kg/min]'].split(',')
				y = int(y[0]) + (0.1)*(int(y[1]))
			return y
		except Exception as e:
			x += ((-1)**(count))*((count+1)*5)
			count += 1
			if count > 20:
				break
	return np.nan

# Get non cruise level fuel consumption
def non_cruise_lvl_fuel(x,z_sign,dfs_climb = bada_data_climb,dfs_descent = bada_data_descent):
	count = 0
	if z_sign == 1 :
		dfs =  dfs_climb
	else:
		dfs =  dfs_descent
	while count>=0:
		try:
			try:
				y = float(dfs.loc[x,'fuel [kg/min]'])
			except Exception as e:
				y = dfs.loc[x,'fuel [kg/min]'].split(',')
				y = int(y[0]) + (0.1)*(int(y[1]))
			return y
		except Exception as e:
			x += ((-1)**(count))*((count+1)*5)
			count += 1
			if count > 20:
				break
	return np.nan



# Initialize all the data
flight_data['start_tas'] = flight_data['start_flightlevel'].apply(lambda x : cruise_lvl_speed(x))
flight_data['start_time'] = flight_data['start_time'].apply(lambda x : datetime.strptime(x, '%H:%M:%S'))
test_2d_plane = ncf_dict['MERGED'][:,:,:,35:]
ncf_file.close()
del ncf_dict
temp = np.empty((5,14,14,31))
temp[0] =  test_2d_plane[0,:,:,:]
temp[1] =  test_2d_plane[1,:,:,:]
temp[2] =  test_2d_plane[1,:,:,:]
temp[3] =  test_2d_plane[2,:,:,:]
temp[4] =  test_2d_plane[2,:,:,:]
# temp = test_2d_plane[:,:,:,35:] # sliced 2d data for global non co2 temp merged.
print("Shape of data loaded {} . \n".format(temp.shape))
# Each unit cell has size 2 degrees on both horizontal and vertical direction


# Calculate L2 distance between points
def cal_dist(a,b):
	# Calculates distance between two given cordinate sets.
	# idx-0 : longitude , idx-1 : latitude
	r = np.linalg.norm(a-b)
	
	return r

# Calculate distance between coordinates
def cal_geo_dist(a,b):
	lat1 = math.radians(a[0])
	lon1 = math.radians(a[1])
	lat2 = math.radians(b[0])
	lon2 = math.radians(b[1])

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2  # Haversine formula
	
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

	return R * c

# Get convex set of points which interesect the voxel blocks based on the given 3d data.
def get_cnv_set(a,b):
	p_list = [0.0]
	state = [0 for _ in range(len(a))]
	for j in range(len(state)):
		if a[j] % 2 == 1:
			state[j] = 1

	no_units = [int(round(np.abs(b[i] - a[i]))) for i in range(len(a))]
	for j in range(len(no_units)):
		for curr_idx in range(state[j],no_units[j],2):
			x_3 = a[j] + (np.sign(b[j]-a[j])*curr_idx)
			x_2 = b[j]
			x_1 = a[j]
			p = (x_3 - x_2) / (x_1 - x_2)
			p_list.append(round(p,3))
	
	if 1.00 not in p_list:
		p_list.append(1.0)
	p = list(set(p_list))
	p_list.sort(reverse=True)
	return p_list

# returns the the voxel index corresponding to the data. 
def get_voxel_pos(a,b):
	# round to nearest odd integer.
	vector_int = np.vectorize(int)
	mid_point = (b + a)/2
	mid_point = vector_int(2*np.floor(mid_point/2)+1)
	return mid_point


# Returns total increase in temp between two points and based on the given temperature profile data.
def get_temp_route(A,B,curr_time,temp=temp,p_list_flag=False):
	# print("shape is {}".format(temp.shape))
	p_list = get_cnv_set(A,B)
	global_temp = 0
	counter = 0
	y_axis = 35
	x_axis = 29
	for i in range(len(p_list)-1):
		y_axis = 35
		x_axis = 29

		x_2 = A*(p_list[i+1]) + B*(1-p_list[i+1])
		x_1 = A*(p_list[i]) + B*(1-p_list[i])
		curr_path_len_z = cal_dist(x_2[-1],x_1[-1])*1*0.304  # to km
		curr_path_len_xy = cal_geo_dist(x_2[:-1],x_1[:-1]) # to km

		curr_voxel_pos_xy = get_voxel_pos(x_2[:-1],x_1[:-1])
		curr_voxel_pos_z = get_voxel_pos(x_2[-1],x_1[-1])

		cruise_fuel_con = cruise_lvl_fuel(int(curr_voxel_pos_z*10))
		cruise_speed = cruise_lvl_fuel(int(curr_voxel_pos_z*10))

		z_sign = np.sign(x_2[-1] - x_1[-1])
		noncruise_fuel_con = non_cruise_lvl_fuel(int(curr_voxel_pos_z*10),z_sign)
		noncruise_speed = non_cruise_lvl_speed(int(curr_voxel_pos_z*10),z_sign)
		# print("{}".format(curr_path_len))
		while True:
			if counter > 3 :
				break
			try:
				xy_time = curr_path_len_xy/cruise_speed
				z_time = curr_path_len_z/noncruise_speed
				curr_time = curr_time + timedelta(minutes= xy_time + z_time)
				global_temp += ((cruise_fuel_con*(xy_time)) + noncruise_fuel_con*(z_time))*\
				(temp[(curr_time.hour//3)-2,int(np.floor((curr_voxel_pos_z - 11)/2)),int((curr_voxel_pos_xy[0]-y_axis)/2),int((curr_voxel_pos_xy[1]+x_axis)/2)])
				break
			except Exception:
				x_axis -= 1
				counter += 1
				pass

	return global_temp, curr_time

# An out of bound check when the flight goes outside the limits.
def out_of_bound(new_pos,new_level):
	if (new_pos[0] > 60) or (new_pos[0] <34):
		return True
	elif (new_pos[1] > 30) or (new_pos[1] < -30 ):
		return True
	elif (new_level > 40) or (new_level <10 ):
		return True
	else:
		return False


# Performs the random walk
def random_walk(A,B,init_flevel,init_time,coeff,step):
	# start_time = datetime.now()
	theta_linspace = np.linspace(-0.436,0.436,15) # value corresponding to beta=25 is 0.436
	phi_linspace = np.linspace(-0.436,0.436,15)
	delta_1 = 0.75
	walk_step_size = step
	num_steps = 200 # Max steps one walker can take.
	path_list = np.hstack((np.array(A,dtype=np.float64),init_flevel))
	curr_pos = np.array(A,dtype=np.float64)

	curr_level = init_flevel
	curr_time = init_time
	global_temp = 0
	for i in range(num_steps):
		if (cal_dist(curr_pos,B) <= 1.2) or (curr_pos[1] > 30):
			# print("found")
			break
		curr_dir = B - curr_pos
		
		curr_slope = curr_dir[0]/curr_dir[1] # Calculate direction/ slope
		
		curr_theta = np.arctan(curr_slope) # Calculate argument

		min_temp_incr = np.Inf
		new_pos = [0,0]
		new_level = curr_level
		theta_out = 0
		phi_out = 0
		for h in range(len(phi_linspace)):
			rand_phi = phi_linspace[h]
			for i in range(len(theta_linspace)):
				rand_theta = theta_linspace[i]
				new_pos[0] = curr_pos[0] + walk_step_size*2*np.sin(curr_theta + rand_theta)*np.cos(rand_phi/coeff)
				new_pos[1] = curr_pos[1] + walk_step_size*2*np.cos(curr_theta + rand_theta)*np.cos(rand_phi/coeff)
				new_level = (((curr_level + walk_step_size*2*np.sin(rand_phi/coeff))))
				# Out of bound check
				if out_of_bound(new_pos,new_level):
					continue
				temp_222, _ = get_temp_route(np.hstack((curr_pos,[curr_level])),np.hstack((new_pos,[new_level])),curr_time)
				if temp_222 <= min_temp_incr:
					min_temp_incr = temp_222
					theta_out = rand_theta
					phi_out = rand_phi
		
		delta_1 = np.random.uniform(-0.35,0.35)
		curr_pos[0] = (curr_pos[0] + walk_step_size*np.sin(curr_theta + theta_out + delta_1))*np.cos(phi_out/coeff)
		curr_pos[1] = (curr_pos[1] + walk_step_size*np.cos(curr_theta + theta_out + delta_1))*np.cos(phi_out/coeff)
		curr_level = curr_level + walk_step_size*np.sin(phi_out/coeff) 
		path_list = np.vstack((path_list,np.hstack((curr_pos,curr_level))))
		temp_temp, temp_time = get_temp_route(path_list[-2],path_list[-1],curr_time)
		global_temp += temp_temp
		curr_time = curr_time

	# print("end time : {}".format((datetime.now() - start_time).total_seconds()))
	# exit()
	return [path_list,global_temp]


# plot 3d charts for each flight
def plot_3d_walks(walks,count_input,A,B,init_flevel,curr_time,coeff,step=2,flag = False):
	try:
		del fig
		del ax
	except Exception as e: pass
	# fig, axes = plt.subplots(1,1,figsize=(12,10),dpi=150)
	fig = plt.figure(figsize=(20,20))
	ax = fig.gca(projection='3d')
	max_len = 0
	z_end = []
	temp_list = []
	count = 0

	for walk in walks:
		count +=1
		x = walk[0][:,0]
		y = walk[0][:,1]
		Z = walk[0][:,2]
		z_end.append(Z[-1])
		temp_list.append(walk[1])
		if len(walks[0]) > max_len:
			max_len = len(walk[0])
		ax.plot3D(x,y,Z)

	ax.scatter(A[0],A[1],init_flevel,s=100,marker='x',label = "START")
	ax.scatter(B[0],B[1],np.mean(z_end),s=100,marker='x',label = "Appx. END")

	ax.scatter([],[],[],label = "Min $\Delta T$ {}e-12".format(np.round(min(temp_list),4)))
	ax.scatter([],[],[],label = "Avg $\Delta T$ {}e-12".format(np.round(np.mean(temp_list),4)))
	ax.scatter([],[],[],label = "Max $\Delta T$ {}e-12".format(np.round(max(temp_list),4)))
	tmp_1 = get_temp_route(np.hstack((A,init_flevel)),np.hstack((B,init_flevel)),curr_time)
	p_list = get_cnv_set(A,B)
	direct_list_x = []
	direct_list_y = []
	for i in range(len(p_list)):
		temp_xy = A*(p_list[i]) + B*(1-p_list[i])

		direct_list_x.append(temp_xy[0])
		direct_list_y.append(temp_xy[1])
	ax.scatter(direct_list_x,direct_list_y,[init_flevel for _ in range(len(p_list))],label = "Direct Path with $\Delta T$ {}e-12".format(np.round(tmp_1[0],4)))

	ax.view_init(15, -10)
	ax.set_xlabel("Latitude")
	ax.set_ylabel("Longitude")
	ax.set_zlabel("Flight Level")
	plt.legend()
	prefix_name = "./3d_walk_plots/" + str(int(coeff*1000)).zfill(4) + "_" +str(int(step*1000)).zfill(4)
	if not os.path.isdir(prefix_name):
		os.mkdir(prefix_name)

	name = prefix_name + "/3d_walk_plots_" + str(count_input).zfill(3)
	plt.title("Weighting in z direction {}. Step size {}.".format(str(int(coeff*1000)).zfill(4),str(int(step*1000)).zfill(4)))
	# plt.show()
	plt.savefig(name)
	del fig, walks, ax, temp_list
	del x,y,Z
	
	# exit()
	return tmp_1[0]


# Main function
def main(coeff,step):
	num_steps = 200 # Max steps one walker can take.
	num_walkers = 15 # total number of walkers for each input
	print("Running random walk with max steps: {} and number of walkers: {}".format(num_steps,num_walkers))
	theta_linspace = np.linspace(-0.436,0.436,6) # max range in xy plane
	phi_linspace = np.linspace(-0.436,0.436,6) # max range in z axis
	delta_1 = 0.75
	# iterate over all given input
	count_input = 0
	avg_temp = []
	min_temp = []
	direct_temp = []
	for inpput_idx in tqdm(range(len(flight_data))):
		if inpput_idx == 41:
			continue
		test_idx  = inpput_idx
		A = np.array([flight_data.loc[test_idx,"start_latitudinal"],flight_data.loc[test_idx,"start_longitudinal"]],dtype=np.float64) # Starting point
		B = np.array([flight_data.loc[test_idx,"end_latitudinal"],flight_data.loc[test_idx,"end_longitudinal"]],dtype=np.float64) # Destination point
		init_time = flight_data.loc[test_idx,"start_time"]
		init_flevel = int(flight_data.loc[test_idx,'start_flightlevel']/10) # Start flight level
		print("start pos alt {} - {} ,End pos: {} ".format(A,init_flevel,B))
		# Perform walks.
		B = np.array(B,dtype=np.float64)
		# walks = [] # For each element in this list, the first element of an element contains the path list and second contains the global temp increase
		curr_level = init_flevel
		num_cores = mp.cpu_count()
		# walk_input = [(A,B,init_flevel,init_time,coeff,step) for _ in range(num_walkers)] # For running without parallel processing.
		walks = Parallel(n_jobs = num_cores)(delayed(random_walk)(A,B,init_flevel,init_time,coeff,step) for _ in range(num_walkers))
		temp_list = [walk[1] for walk in walks]
		tmp_direct = plot_3d_walks(walks,count_input,A,B,init_flevel,init_time,coeff,step) # Store direct path increae in temperature.
		temp_list.append(tmp_direct)
		avg_temp.append(np.mean(temp_list))
		min_temp.append(min(temp_list))

		direct_temp.append(tmp_direct)
		count_input += 1
		del walks
		del temp_list
		print("Running : Avg temp: {} , Min temp: {} , Direct temp: {} \n".format(sum(avg_temp),sum(min_temp),sum(direct_temp)))
		print("XXXXXXX")


	return [sum(avg_temp),sum(min_temp),sum(direct_temp)]



# calling funtion when run through command line.
if __name__ == "__main__":
	coe_lins = [2.895]
	step_lins = [2.657]
	# From the above paramters, plots will be created and for each flight with corresponding walks.

	for coe in coe_lins:
		for step in step_lins:
			temp_var = main(coe,step)
			try:
				del fig 
			except Exception as e: pass
			fig, axes = plt.subplots(1,1,figsize=(8,6),dpi=150)
			plt.subplot(111)
			plt.bar(["Min"],temp_var[1]*1e-12, align='center',label="Min temp", width = 0.85,color="teal")
			plt.bar(["Direct"],temp_var[2]*1e-12, align='center',label="Direct temp", width = 0.85,color='darkslateblue')
			plt.xlabel("Path type")
			plt.ylabel(" $\Delta T$ ")
			plt.legend()
			fig_name = "bar_Chart_" + str(int(coe*1000)).zfill(4) + "_" +str(int(step*1000)).zfill(4)
			plt.savefig(fig_name)

			print("Avg temp: {} , Min temp: {} , Direct temp: {}".format(temp_var[0],temp_var[1],temp_var[2]))





