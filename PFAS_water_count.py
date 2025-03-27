import sys
import numpy as np
import mdtraj as md
import sys
import os
from multiprocessing import Pool


def PFAS_to_dataframe(nth_dup, system_name):
    traj=traj_list[nth_dup]
    topo = traj.topology
    UNL_resID_list=[]
    for i in range(topo.n_residues):
        match=re.findall("UNL", str(topo.residue(i)) )
        if match == ["UNL"] :
            UNL_resID_list.append(i)                     #### This patch of code creates a list of the residue IDs of the PFAS/UNL residues
    ##### compute the COMS
    COM_list=[]
    for i in UNL_resID_list:                
        selection = topo.select("resid "+str(i))
        new_traj = traj.atom_slice(selection)
        COM = md.compute_center_of_mass(new_traj)
        COM_list.append(COM)

    PFAS_frames_COMxyz= np.array(COM_list)      #makes an array of dimensions no_of_PFAS x no_of_frames x 3 i.e. stores the xyz coordinates of the com of the 100 PFAS molecules across all frames
    shp= PFAS_frames_COMxyz.shape
    n_steps=shp[1]
    n_mols= shp[0]
    stp=PFAS_frames_COMxyz[:,0,:]
    tlist_nPFAS_r=[]      # this is a list of 1D arrays where each array represents a timestep and contain the number of PFAS molecules within the r
    for j in range(n_steps):
        step_j=PFAS_frames_COMxyz[:,j,:]
        nPFAs_r_lst=[]
        r = 1.5
        while r <= 5.1 :
            resindx_zcom_ls=[] 
            for k in range(n_mols):
                z_crd=step_j[k,2]
                if abs(z_crd - 7.0) < r:      # Calculating the number of PFAS within 1.5nm from the graphene
                    resindx_zcom_ls.append((UNL_resID_list[k],z_crd))
            nPFAs_r_lst.append(len(resindx_zcom_ls))
            r += 0.1
        nPFAs_r_arr=np.array(nPFAs_r_lst)
        tlist_nPFAS_r.append(nPFAs_r_arr)
    np.savetxt(str(nth_dup+1)+system_name+'tstp_nPFAS_r'+'.txt', tlist_nPFAS_r, delimiter =" ",header='')        
    return nth_dup

def water_to_dataframe(nth_dup, system_name):
    traj=traj_list[nth_dup]
    WAT=traj.top.select("resname == 'HOH' and symbol == 'O'")      # an array containing the residue indices of water
    fr_XYZ_WAT=traj.xyz[:,WAT[0],:]
    for i in WAT[1:]:
        y=traj.xyz[:, i, :]
        fr_XYZ_WAT= np.dstack((fr_XYZ_WAT,y))          #so we stack the xyz coordinates of each of the water O 
    
    print(fr_XYZ_WAT.shape)
    shp_w= fr_XYZ_WAT.shape
    n_steps_w=shp_w[0]
    n_mols_w= shp_w[2]
    stp_w=fr_XYZ_WAT[0,:,:]
    tlist_WAT_r=[]      # this is a list of 1D arrays where each array represents a timestep and contain the number of water molecules within the r
    for j in range(n_steps_w):
        step_j=fr_XYZ_WAT[j,:,:]
        nWAT_r_lst=[]
        r = 1.5
        while r <= 5.1 :
            wat_atmindx_zcrd_ls=[] 
            for k in range(n_mols_w):
                z_crd=step_j[2,k]
                if abs(z_crd - 7.0) < r:      # Calculating the number of PFAS within 1.5nm from the graphene
                    wat_atmindx_zcrd_ls.append((WAT[k],z_crd))
            nWAT_r_lst.append(len(wat_atmindx_zcrd_ls))
            r += 0.1
        nWAT_r_arr=np.array(nWAT_r_lst)
        tlist_WAT_r.append(nWAT_r_arr)
    np.savetxt(str(nth_dup+1)+system_name+'tstp_WATER_r'+'.txt', tlist_WAT_r, delimiter =" ",header='')


def read_in_traj(top_name,i):
    traj_i = md.load('800frames_PFBA_'+str(i)+'.dcd',top=top_name)
    print("the traj_being read is :"+str(i))
    return(traj_i)



system_name = 'TFEMA_PFBA_'
top_name = 'TFEMA_PFBA.prmtop'
traj_list=[]

with Pool(10) as p:
    args = [(top_name,i) for i in range(10)]
    traj_list = p.starmap(read_in_traj, args)         ##### the list returned here will be the list of trajectories you read in
    p.close()
    p.join()

with Pool(10) as p:
    args = [(nth_dup,system_name) for nth_dup in range(10)]
    p.starmap(PFAS_to_dataframe, args)
    p.close()
    p.join()

with Pool(10) as p:
    args = [(nth_dup,system_name) for nth_dup in range(10)]
    p.starmap(water_to_dataframe, args)
    p.close()
    p.join()
