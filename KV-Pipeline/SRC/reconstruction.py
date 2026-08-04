"""
Reconstruction Module
Provides a wrapper over various TIGRE algorithms (FDK, SIRT, CGLS, etc.).
Automatically applies filters and limits.
"""
import tigre
import numpy as np
from tigre.utilities.im3Dnorm import im3DNORM
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from tigre.utilities.flatten_detector import flatten_detector, unflatten_detector
import os
import time

# Import the new overlapping chunks reconstructor
from chunk_FDK import chunked_fdk

def all_algorithms(geo, angles, proj, recon_name, final_imag_folder_path, is_plotting, init=None, niter=20):
    """
    Executes the specified reconstruction algorithm based on a string mapping.
    Includes both analytical (FDK) and iterative (CGLS, SART) methods.
    """
    lmbda = 1
    lambdared = 0.9999
    verbose = False
    qualmeas = ["RMSE", "SSD"]
    blcks = 10
    order = "random"

    alpha = 0.002
    ratio = 0.94
    ng = 25

    startTime = time.time()

    # Route to the appropriate TIGRE algorithm
    if recon_name == "FDK":
        img = algs.fdk(proj, geo, angles, verbose=verbose, filter="hann")
    elif recon_name == "ChunkFDK":
        # Processes long conveyor scans in overlapping stitched windows
        img = chunked_fdk(proj, geo, angles, chunk_size_idx=44, overlap_size_idx=24, verbose=verbose)
    elif recon_name == "SIRT":
        img = algs.sirt(proj, geo, angles, niter, lmbda=lmbda, lmbda_red=lambdared, verbose=verbose)
    elif recon_name == "SART":
        img = algs.sart(proj, geo, angles, niter, lmbda=lmbda, lmbda_red=lambdared, verbose=verbose)
    elif recon_name == "OS-SART":
        img = algs.ossart(proj, geo, angles, niter, lmbda=lmbda, lmbda_red=lambdared, verbose=verbose, blocksize=blcks, OrderStrategy=order)
    elif recon_name == "CGLS":
        img = algs.cgls(proj, geo, angles, niter, init=init, verbose=verbose)
    elif recon_name == "LSQR":
        img = algs.lsqr(proj, geo, angles, niter, verbose=verbose)
    elif recon_name == "LSMR":
        img = algs.lsmr(proj, geo, angles, niter, lmbda=0, verbose=verbose)
    elif recon_name == "hLSQR":
        img = algs.hybrid_lsqr(proj, geo, angles, niter, verbose=verbose)
    elif recon_name == "AB-GMRES":
        img = algs.ab_gmres(proj, geo, angles, niter, verbose=verbose)
    elif recon_name == "BA-GMRES":
        img = algs.ba_gmres(proj, geo, angles, niter, verbose=verbose)
    elif recon_name == "AB-GMRES-FDK-back":
        img = algs.ab_gmres(proj, geo, angles, niter, backprojector="FDK", verbose=verbose)
    elif recon_name == "BA-GMRES-FDK-back":
        img = algs.ba_gmres(proj, geo, angles, niter, backprojector="FDK", verbose=verbose)
    elif recon_name == "ASD-POCS":
        noise_projections = CTnoise.add(proj, Poisson=1e5, Gaussian=np.array([0, 10]))
        epsilon = (im3DNORM(tigre.Ax(algs.fdk(noise_projections, geo, angles), geo, angles) - noise_projections, 2) * 0.15)
        img = algs.asd_pocs(proj, geo, angles, niter, tviter=ng, maxl2err=epsilon, alpha=alpha, lmbda=lmbda, lmbda_red=lambdared, rmax=ratio, verbose=verbose)
    elif recon_name == "OS_ASD_POCS":
        noise_projections = CTnoise.add(proj, Poisson=1e5, Gaussian=np.array([0, 10]))
        epsilon = (im3DNORM(tigre.Ax(algs.fdk(noise_projections, geo, angles), geo, angles) - noise_projections, 2) * 0.15)
        img = algs.os_asd_pocs(proj, geo, angles, niter, tviter=ng, maxl2err=epsilon, alpha=alpha, lmbda=lmbda, lmbda_red=lambdared, rmax=ratio, verbose=verbose, blocksize=10)
    elif recon_name == "AwASD_POCS":
        noise_projections = CTnoise.add(proj, Poisson=1e5, Gaussian=np.array([0, 10]))
        epsilon = (im3DNORM(tigre.Ax(algs.fdk(noise_projections, geo, angles), geo, angles) - noise_projections, 2) * 0.15)
        img = algs.awasd_pocs(proj, geo, angles, niter, tviter=ng, maxl2err=epsilon, alpha=alpha, lmbda=lmbda, lmbda_red=lambdared, rmax=ratio, verbose=verbose, delta=np.array([-0.005]))
    elif recon_name == "IRN-CGLS-TV":
        img = algs.irn_tv_cgls(proj, geo, angles, niter, lmbda=5, niter_outer=2)

    endTime = time.time()
    print("Time for reconstruction : ", endTime - startTime)

    if is_plotting and final_imag_folder_path:
        os.makedirs(final_imag_folder_path, exist_ok=True)
        tigre.plotimg(img, dim="Z", savegif=os.path.join(final_imag_folder_path, f"{recon_name}.gif"))
        np.save(os.path.join(final_imag_folder_path, f"{recon_name}.npy"), img)

    return img