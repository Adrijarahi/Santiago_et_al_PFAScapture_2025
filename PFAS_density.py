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
    print(stp.shape)
    tlist_nPFAS_r=[]      # this is a list of 1D arrays where each array represents a timestep and contain the number of PFAS molecules within the r
    for j in range(n_steps):
        step_j=PFAS_frames_COMxyz[:,j,:]
        #print(step_j.shape)
        nPFAs_r_lst=[]
        r = 0.0
        while r <= 7.0 :
            resindx_zcom_ls=[] 
            for k in range(n_mols):
                z_crd=step_j[k,2]
                if abs(z_crd - 7.0) < r:      # Calculating the number of PFAS within 1.5nm from the graphene
                    resindx_zcom_ls.append((UNL_resID_list[k],z_crd))
            nPFAs_r_lst.append(len(resindx_zcom_ls))
            r += 0.1
        nPFAs_r_arr=np.array(nPFAs_r_lst)
        tlist_nPFAS_r.append(nPFAs_r_arr)
    np.savetxt(str(nth_dup+1)+system_name+'tstp_nPFAS_z0_z7'+'.txt', tlist_nPFAS_r, delimiter =" ",header='')        