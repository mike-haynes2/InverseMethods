import numpy as np
import scipy.special as sp
from gaussxw import gaussxw
import matplotlib as mpl
from scipy import constants
from pathlib import Path
from matplotlib import cm

def juno_field_vector(phi, theta, L):
    ## phi must be input in the RIGHT-HANDED system (i.e., East longitude)
    ## so for the magnetic equator at 110 degree system III WEST longitude,
    ## input phi = (360-110) = 250 * (pi/180). MUST BE CONVERTED TO RADIANS. theta is measured from the spin axis:
    ## theta = 0: north pole, theta = pi/2: equator, theta = pi: north pole
    ## L is L shell, given in R_J (i.e., for Europa, L = 9.38)

    def dipole_field_global_cartesian(x,y,z):

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)

        g_1_0 = 410993.4e-9 # nT
        g_1_1 = -71305.9e-9
        h_1_1 = 20958.4e-9

        a = 71492e3

        r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        theta_hat = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-1*np.sin(theta)])
        phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(np.shape(phi))])

        # unit vector always magnitude 1
        #r_hat_mag = np.sqrt(r_hat[0]**2 + r_hat[1]**2 + r_hat[2]**2)

        Br = 2*(a/r)**3*(g_1_0*np.cos(theta) + np.sin(theta)*(g_1_1*np.cos(phi) + h_1_1*np.sin(phi)))
        B_theta = -(a/r)**3*(-g_1_0*np.sin(theta) + np.cos(theta)*(g_1_1*np.cos(phi) + h_1_1*np.sin(phi)))
        B_phi = -(a/r)**3*(-g_1_1*np.sin(phi) + h_1_1*np.cos(phi))

        r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        theta_hat = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-1*np.sin(theta)])
        phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(np.shape(phi))])

        B = Br*r_hat + B_theta*theta_hat + B_phi*phi_hat

        Bmag = np.linalg.norm(B)

        return B, Bmag

    def quadrupole_field_global_cartesian(x,y,z):

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)

        g_2_0 = 11796.7e-9 # nT
        g_2_1 = -56972.4e-9
        h_2_1 = -42549e-9
        g_2_2 = 48250.2e-9
        h_2_2 = 20221.5e-9


        P_2_0 = 0.5*(3*np.cos(theta)**2-1)
        P_2_1 = 3/np.sqrt(3)*np.sin(theta)*np.cos(theta)
        P_2_2 = 3/np.sqrt(12)*np.sin(theta)**2

        a = 71492e3

        r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        theta_hat = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-1*np.sin(theta)])
        phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(np.shape(phi))])

        r_hat_mag = np.sqrt(r_hat[0]**2 + r_hat[1]**2 + r_hat[2]**2)

        Br = 3*(a/r)**4*(P_2_0*g_2_0 
                         + P_2_1*(g_2_1*np.cos(phi) + h_2_1*np.sin(phi)) 
                         + P_2_2*(g_2_2*np.cos(2*phi) + h_2_2*np.sin(2*phi)))

        B_theta = -(a/r)**4*(-3*g_2_0*np.cos(theta)*np.sin(theta) 
                            + 3/np.sqrt(3)*np.cos(2*theta)*(g_2_1*np.cos(phi) + h_2_1*np.sin(phi))
                            + 3/np.sqrt(3)*np.sin(theta)*np.cos(theta)*(g_2_2*np.cos(2*phi) + h_2_2*np.sin(2*phi)))

        B_phi = -(a/r)**4*(3/np.sqrt(3)*np.cos(theta)*(-g_2_1*np.sin(phi) + h_2_1*np.cos(phi))
                          + 3/np.sqrt(3)*np.sin(theta)*(-g_2_2*np.sin(2*phi) + h_2_2*np.cos(2*phi)))

        r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        theta_hat = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-1*np.sin(theta)])
        phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(np.shape(phi))])

        B = Br*r_hat + B_theta*theta_hat + B_phi*phi_hat

        Bmag = np.linalg.norm(B)

        return B, Bmag

    def magnetodisc_integral_cartesian(x,y,z, a, deg):

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)

        def coord_transform(r, theta, phi):

            x = r*np.cos(phi)*np.sin(theta)
            y = r*np.sin(phi)*np.sin(theta)
            z = r*np.cos(theta)
            # this function takes in x,y,z coordinates in the geographic, system III longitude system, S

            # first, we convert the geographic x,y,z coordinates into a system with xy plane in the magnetic equator

            theta_D = 9.3*np.pi/180 # tilt angle of the magnetodisc

            phi_D = 2*np.pi - 204*np.pi/180 # azimuthal angle of the normal to the magnetodisc (at this point, magnetic and rotational equators coincide)

            x_plane_normal = np.cos(phi_D)*np.sin(theta_D)
            y_plane_normal = np.sin(phi_D)*np.sin(theta_D)
            z_plane_normal = np.cos(theta_D)

            normal_vec = np.array([x_plane_normal, y_plane_normal, z_plane_normal])

            normal_vec_mag = np.linalg.norm(normal_vec)

            normal_unit_vec = normal_vec/normal_vec_mag

            # draw the vector from the point to the plane

            d = np.abs(normal_unit_vec[0]*x + normal_unit_vec[1]*y + normal_unit_vec[2]*z)

            d_vec = d*normal_vec

            # now, we need to determine whether the point is "above" or "below" the plane
            # to do this, take the dot product of the normal vector and the vector the the point
            # if this is positive, the point is above the plane, otherwise it is below

            sign_check = np.dot(normal_vec, np.array([x,y,z]))

            if sign_check > 0:
                sign = 1
            else:
                sign = -1

            # now we can determine z (in the tilted system)

            z_cy = sign*d

            # projection of given point onto the plane
            P = np.array([x-sign*d_vec[0], y-sign*d_vec[1], z-sign*d_vec[2]])

            rho = np.linalg.norm(P)

            # plot the vector from the origin to P, THIS IS THE RHO VECTOR!

            rho_vec = np.array([P[0],P[1],P[2]])

            rho_hat = rho_vec/np.linalg.norm(rho_vec)

            z_hat = normal_unit_vec

            phi_hat = np.cross(z_hat, rho_hat)

            return rho, z_cy, rho_hat, z_hat, phi_hat

        # Function to evaluate integral for B_rho with Gaussian quadrature
        def Gauss_B_rho(N, start, stop, rho, z, a, D):

            if np.abs(z) > D:
                def f(l):
                    # equation 14 from Con81 if outside the sheet
                    bes1 = sp.jv(1, l*rho)
                    bes2 = sp.jv(0, l*a)
                    func = np.sign(z)*bes1*bes2*np.sinh(l*D)*np.exp(-l*np.abs(z))/l
                    return func
            else:
                def f(l):
                    # equation 17 from Con81 if inside the sheet
                    bes1 = sp.jv(1, l*rho)
                    bes2 = sp.jv(0, l*a)
                    func = bes1*bes2*np.sinh(l*z)*np.exp(-l*D)/l
                    return func

            my_sum = 0

            l, w = gaussxw(N+1)

            l_prime = l*0.5*(stop-start)+0.5*(stop+start)

            w_prime = w*0.5*(stop-start)

            for k in range (N):

                my_sum += w_prime[k]*f(l_prime[k])

            return my_sum

        def Gauss_B_z(N, start, stop, rho, z, a, D):

            if np.abs(z) > D:
                def f(l):
                    # equation 15 from Con81 if outside the sheet
                    bes1 = sp.jv(0, l*rho)
                    bes2 = sp.jv(0, l*a)
                    func = bes1*bes2*np.sinh(l*D)*np.exp(-l*np.abs(z))/l
                    return func
            else:
                def f(l):
                    # equation 18 from Con81 if inside the sheet
                    bes1 = sp.jv(0, l*rho)
                    bes2 = sp.jv(0, l*a)
                    func = bes1*bes2*(1-np.exp(-l*D)*np.cosh(l*z))/l
                    return func

            my_sum = 0

            l, w = gaussxw(N+1)

            l_prime = l*0.5*(stop-start)+0.5*(stop+start)

            w_prime = w*0.5*(stop-start)

            for k in range (N):

                my_sum += w_prime[k]*f(l_prime[k])

            return my_sum

        R_J = 71492e3

        D = 3.6 * R_J #RJ
        mu_0_I_over_2 = 139.6*1e-9 # nT

        rho, z, rho_hat, z_hat, phi_hat = coord_transform(r,theta,phi)

        ########## Integrals #######################

        # Integrals are performed by dividing the integral into sub-integrals,
        # defined by the zeros of the J1 (?) bessel function. Each sub-integral
        # is calculated, and if the contribution it makes to the total integral
        # is below the tolerance, the integral is considered to have converged.
        # Otherwise, the sub-integral is added to the sum

        N_gauss = deg #degree (?) of the gaussian quadrature

        count = 0

        tol = 1 # error tolerance, in nT

        diff = tol+1   

        #### First Integrate for B_rho ########
        B_rho = 0 # initial value of the integrals

        while diff > tol:
            zeros= sp.jn_zeros(1,count+1)/rho
            if count == 0:
                start = 0 # lower bound of the sub-integral, set to zero (?) for the first iteration
            else:
                start = zeros[count-1]

            stop = zeros[count] # upper bound of the sub-integral

            count += 1

            if count%10 == 0:
                print(count)

            val = Gauss_B_rho(N_gauss, start, stop, rho, z, a, D)

            B_rho += mu_0_I_over_2*2*val

            diff = mu_0_I_over_2*2*val

        #########################

        #### Now Integrate for B_z ########

        count = 0

        diff = tol+1

        B_z = 0 # initial value of the integrals

        while diff > tol:
            zeros= sp.jn_zeros(0,count+1)/rho
            if count == 0:
                start = 0 # lower bound of the sub-integral, set to zero (?) for the first iteration
            else:
                start = zeros[count-1]

            stop = zeros[count] # upper bound of the sub-integral

            count += 1

            val = Gauss_B_z(N_gauss, start, stop, rho, z, a, D)

            B_z += mu_0_I_over_2*2*val

            diff = mu_0_I_over_2*2*val

        #########################

        ############################################

        irho = 16.7*1e-9

        B_phi = 2.7975*R_J*irho/rho

        if np.abs(z) < D:
            B_phi = B_phi*np.abs(z)/D
        if z > 0:
            B_phi = -1*B_phi

        # rho_hat = np.array([np.cos(phi), np.sin(phi), 0]) # cylindrical basis vectors in cartesian coordinates
        # phi_hat = np.array([-1*np.sin(phi), np.cos(phi), 0])
        # z_hat = np.array([0,0,1])

        #B_cart = B_rho*rho_hat + B_z*z_hat # convert cylindrical field components to cartesian coordinates
        Bx = B_rho*rho_hat[0] + B_z*z_hat[0] + B_phi*phi_hat[0]
        By = B_rho*rho_hat[1] + B_z*z_hat[1] + B_phi*phi_hat[1]
        Bz = B_rho*rho_hat[2] + B_z*z_hat[2] + B_phi*phi_hat[2]

        B_cart = np.array([Bx, By, Bz])

        return B_cart, np.linalg.norm(B_cart)

    def get_field(x,y,z, model = 'dipole_quadrupole_magnetodisc', deg = 50):

        R_J = 71492e3
        R0 = 7.8*R_J
        R1 = 51.4*R_J

        B_dip, Bmag = dipole_field_global_cartesian(x,y,z)
        B_quad, Bmag = quadrupole_field_global_cartesian(x,y,z)

        if model == 'dipole':
            B = B_dip
            Bmag = np.linalg.norm(B_dip)

        if model == 'dipole_quadrupole':
            B = B_dip + B_quad
            Bmag = np.linalg.norm(B_dip)

        if model == 'dipole_quadrupole_magnetodisc':
            B_md_inner, Bmag = magnetodisc_integral_cartesian(x,y,z, R0, deg)
            B_md_outer, Bmag = magnetodisc_integral_cartesian(x,y,z, R1, deg)
            B_md = B_md_inner - B_md_outer
            B = B_dip + B_quad + B_md
            Bmag = np.linalg.norm(B_dip)

        return B, Bmag


    R_J = 71492e3

    x = L*R_J*np.cos(phi)*np.sin(theta)
    y = L*R_J*np.sin(phi)*np.sin(theta)
    z = np.cos(theta)

    x_hat = np.array([np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -1*np.sin(phi)])
    y_hat = np.array([np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)])
    z_hat = np.array([np.cos(theta), -1*np.sin(theta), 0])

    my_B_vec, Bmag = get_field(x, y, z, deg = 1000)

    Bx = my_B_vec[0]
    By = my_B_vec[1]
    Bz = my_B_vec[2]

    B_spherical = Bx*x_hat + By*y_hat + Bz*z_hat

    Br = B_spherical[0]
    Btheta = B_spherical[1]
    Bphi = B_spherical[2]    

    Bx_ephio = Bphi
    By_ephio = -1*Br
    Bz_ephio = -1*Btheta

    #print('B = [%.3f, %.3f, %.3f] nT' % (Bx_ephio*1e9, By_ephio*1e9, Bz_ephio*1e9))
    return Bx_ephio, By_ephio, Bz_ephio
    


## TO CALL: juno_field_vector( (360 - sys3 longitude), (theta), (L-shell) )

# Bx, By, Bz = juno_field_vector((360-110)*np.pi/180, np.pi/2, 26.09)

# print(Bx*1e9, By*1e9, Bz*1e9)