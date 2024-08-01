INCLUDE "dictionary_module.f95"
INCLUDE "properties.f95"
INCLUDE "transmissibility.f95"

MODULE Residual_Jacobian_triplet

    !*****************************************************************!
    ! This module is to evaluate two phase flow residual AND jacobian !
    !*****************************************************************!

    USE Properties
    USE Transmissibility
    USE dictionary_module

    IMPLICIT NONE
    SAVE
    PRIVATE
    PUBLIC :: get_residual_jacobian_triplet !, get_residual, get_jacobian

    CONTAINS

    !SUBROUTINE get_residual(X, P_holder, Sw_holder, Tgeo, K, &
    !    cell_connections, N_cell_connections,N_cells, N_steps, Property_Dict, Well_Loc, Prev_t_step, JAC, RES)

    !SUBROUTINE get_jacobian(X, P_holder, Sw_holder, Tgeo, K, &
    !    cell_connections, N_cell_connections,N_cells, N_steps, Property_Dict, Well_Loc, Prev_t_step, JAC, RES)

    SUBROUTINE get_residual_jacobian_triplet(X, P_holder, Sw_holder, Tgeo, K, &
                    cell_connections, N_cell_connections,N_cells, N_steps, Property_Dict, Well_Loc, Prev_t_step, JAC, RES, IDX_MAT)

        ! Input
        INTEGER(KIND=4), INTENT(IN) :: N_cell_connections, N_cells, N_steps
        REAL(KIND=8), DIMENSION(N_cells*2), INTENT(IN) :: X
        REAL(KIND=8), DIMENSION(N_steps, N_cells), INTENT(IN) :: P_holder, Sw_holder
        REAL(KIND=8), DIMENSION(N_cell_connections), INTENT(IN) :: Tgeo
        REAL(KIND=8), DIMENSION(N_cells), INTENT(IN) :: K
        REAL(KIND=8), DIMENSION(2, N_cells), INTENT(IN) :: Well_Loc
        INTEGER(KIND=4), DIMENSION(N_cells*2, N_cells*2), INTENT(IN) :: IDX_MAT ! TO COMPRESS JACOBIAN MATRIX INTO JACOBIAN VECTOR
        INTEGER(KIND=4), DIMENSION(N_cell_connections, 2), INTENT(IN) :: cell_connections
        ! Output
        REAL(KIND=8), DIMENSION(N_cells*2), INTENT(OUT) :: RES
        REAL(KIND=8), DIMENSION((2*N_cell_connections + N_cells)*4), INTENT(OUT) :: JAC
        ! Local variables
        REAL(KIND=8), DIMENSION(N_cells) :: P, Sw
        REAL(KIND=8), DIMENSION(N_cells) :: poro, rhoo, kro, visc_o,mobo, rhow, krw, visc_w, mobw
        REAL(KIND=8), DIMENSION(N_cells) :: dporodp, drhodp_o, dviscdp_o, dKrodSw, drhodp_w, dviscdp_w, dKrwdSw
        REAL(KIND=8), DIMENSION(N_cells) :: dmobodp, dmobods, dmobwdp, dmobwds
        INTEGER(KIND=4), DIMENSION(N_cell_connections) :: upwinding
        REAL(KIND=8), DIMENSION(N_cell_connections) :: Thalf_o, Thalf_w, F_c_o, F_c_w
        REAL(KIND=8), DIMENSION(N_cell_connections) :: dTdp_o, dTdp_w, dTds_o, dTds_w
        REAL(KIND=8), DIMENSION(N_cell_connections, 2) :: dFdp_o, dFdp_w, dFds_o, dFds_w
        TYPE(dictionary_type) :: Property_Dict
        REAL(KIND=8) :: rho_o_std, rho_w_std, Cr, Co, Cw, Pref, phi
        REAL(KIND=8) :: Swmin, Somin, Kro_, Krw_
        REAL(KIND=8) :: dx, dy, dz, dt
        REAL(KIND=8) :: c1, c2, c3
        REAL(KIND=8) :: BHP, Qinj, rw, re, S, WI, well_1, well_2
        INTEGER(KIND=4) :: i, Prev_t_step
        INTEGER(KIND=8) :: c_1, c_2
        REAL(KIND=8), DIMENSION(N_cells) :: R_o, R_w, F_o, F_w, A_o, A_o_coef, A_w, A_w_coef, Q_o, Q_w
        REAL(KIND=8), DIMENSION(N_cells) :: dA_odp, dA_o_coefdp, dA_wdp, dA_w_coefdp, dQ_odp, dQ_wdp
        REAL(KIND=8), DIMENSION(N_cells) :: dA_ods, dA_wds, dQ_ods, dQ_wds
        JAC = 0.0
        RES = 0.0

        c1    = get_from_dictionary(Property_Dict, "c1"   )
        c2    = get_from_dictionary(Property_Dict, "c2"   )
        c3    = get_from_dictionary(Property_Dict, "c3"   )
        rho_o_std    = get_from_dictionary(Property_Dict, "rho_o_std"   )
        rho_w_std    = get_from_dictionary(Property_Dict, "rho_w_std"   )
        Cr    = get_from_dictionary(Property_Dict, "Cr"   )
        Co    = get_from_dictionary(Property_Dict, "Co"   )
        Cw    = get_from_dictionary(Property_Dict, "Cw"   )
        Pref  = get_from_dictionary(Property_Dict, "Pref" )
        phi   = get_from_dictionary(Property_Dict, "phi"  )
        Swmin = get_from_dictionary(Property_Dict, "Swmin")
        Somin = get_from_dictionary(Property_Dict, "Somin")
        Kro_  = get_from_dictionary(Property_Dict, "Kro_" )
        Krw_  = get_from_dictionary(Property_Dict, "Krw_" )
        dx    = get_from_dictionary(Property_Dict, "dx"   )
        dy    = get_from_dictionary(Property_Dict, "dy"   )
        dz    = get_from_dictionary(Property_Dict, "dz"   )
        dt    = get_from_dictionary(Property_Dict, "dt"   )
        BHP   = get_from_dictionary(Property_Dict, "BHP"  )
        Qinj  = get_from_dictionary(Property_Dict, "Qinj" )
        rw    = get_from_dictionary(Property_Dict, "rw"   )
        re    = get_from_dictionary(Property_Dict, "re"   )
        S     = get_from_dictionary(Property_Dict, "S"    )
        WI    = get_from_dictionary(Property_Dict, "WI"   )
        well_1= get_from_dictionary(Property_Dict, "well_1"    )
        well_2= get_from_dictionary(Property_Dict, "well_2"    )

        P = X(:N_cells)
        Sw = X(N_cells+1 :)

        poro      = porosity(P, Cr, Pref, phi)
        dporodp   = DporosityDp(P, Cr, Pref, phi)
        ! Oil Properties and Their Derivatives w.r.t p and s
        rhoo      = rho_o(P, Co, Pref, rho_o_std)
        kro       = Kr_o(Sw, Somin, Swmin, Kro_)
        visc_o    = mu_o(P, Co, Pref)
        mobo      = (rhoo*kro)/(visc_o*rho_o_std)

        drhodp_o  = Drho_oDp(P, Co, Pref, rho_o_std)
        dviscdp_o = Dmu_oDp(P, Co, Pref)
        dKrodSw   = DKr_oDs(Sw, Somin, Swmin, Kro_)
        dmobodp   = (kro/rho_o_std)*(rhoo*(-1/visc_o**2)*dviscdp_o  + (1/visc_o)*drhodp_o)
        dmobods   = (dKrodSw*rhoo/(visc_o*rho_o_std))
        
        ! Water Properties and Their Derivatives w.r.t p and s
        rhow      = rho_w(P, Cw, Pref, rho_w_std)
        krw       = Kr_w(Sw, Somin, Swmin, Krw_)
        visc_w    = mu_w(P, Cw, Pref)
        mobw      = (rhow*krw)/(visc_w*rho_w_std)

        drhodp_w  = Drho_wDp(P, Cw, Pref, rho_w_std)
        dviscdp_w = Dmu_wDp(P, Cw, Pref)
        dKrwdSw   = DKr_wDs(Sw, Somin, Swmin, Krw_)
        dmobwdp   = (krw/rho_w_std)*(rhow*(-1/visc_w**2)*dviscdp_w  + (1/visc_w)*drhodp_w)
        dmobwds   = (dKrwdSw*rhow/(visc_w*rho_w_std))

        !!!!!========================= RESIDUAL CALCULATION =========================!!!!!

        !!!!!! SOURCE & SINK TERM !!!!!
        Q_o = c1 * WI * mobo * (P - BHP)*Well_Loc(2, :)
        Q_w = c1 * WI * mobw * (P - BHP)*Well_Loc(2, :) + c3* (Qinj*rhow/rho_w_std)*Well_Loc(1, :)

        !!!!!! ACCUMILATION TERM !!!!!
        A_o_coef = (dx*dy*dz*poro*rhoo/(rho_o_std*dt))
        A_o = A_o_coef*( (1 - Sw)*(Co + Cr)*(P - P_holder(Prev_t_step, :)) - (Sw-Sw_holder(Prev_t_step,:)))
        
        A_w_coef = (dx*dy*dz*poro*rhow/(rho_w_std*dt))
        A_w = A_w_coef*(Sw*(Cw + Cr)*(P - P_holder(Prev_t_step, :)) + (Sw - Sw_holder(Prev_t_step, :))) 
        
        !!!!! FLUX TERM !!!!!
        upwinding = up(P, cell_connections, N_cell_connections)

        Thalf_o = c1 * T_full(kro, visc_o, rhoo, rho_o_std, Tgeo, cell_connections, N_cell_connections, N_cells, upwinding)
        Thalf_w = c1 * T_full(krw, visc_w, rhow, rho_W_std, Tgeo, cell_connections, N_cell_connections, N_cells, upwinding)

        F_c_o = Thalf_o*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))
        F_c_w = Thalf_w*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))

        F_o = 0
        F_w = 0

        DO i = 1, N_cell_connections
            F_o(cell_connections(i, 1)) = F_o(cell_connections(i, 1)) + F_c_o(i)
            F_o(cell_connections(i, 2)) = F_o(cell_connections(i, 2)) - F_c_o(i)

            F_w(cell_connections(i, 1)) = F_w(cell_connections(i, 1)) + F_c_w(i)
            F_w(cell_connections(i, 2)) = F_w(cell_connections(i, 2)) - F_c_w(i)
        END DO
        
        R_o = F_o - A_o - Q_o ! Oil Phase Residual
        R_w = F_w - A_w - Q_w ! Water Phase Residual
        !PRINT *, F_o
        !PRINT *, F_w
        RES = RESHAPE([R_o, R_w], [N_cells*2])

        !!!!!========================= JACOBIAN CALCULATION =========================!!!!!

        !!!!! ====DERIVATIVE OF FLUX TERM==== !!!!!

        ! Calculate Derivatives of Transmissibility
        CALL dTdp_dTds(kro, visc_o, rhoo, rho_o_std, dKrodSw, dviscdp_o, drhodp_o, Tgeo, &
                                                        cell_connections, N_cell_connections, N_cells, upwinding, dTdp_o, dTds_o)
        CALL dTdp_dTds(krw, visc_w, rhow, rho_w_std, dKrwdSw, dviscdp_w, drhodp_w, Tgeo, &
                                                        cell_connections, N_cell_connections, N_cells, upwinding, dTdp_w, dTds_w)


        ! Derivative of Oil Phase Flux Term W.R.T p
        dFdp_o = RESHAPE([-Thalf_o + c1 *dTdp_o*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*upwinding, &
            Thalf_o + c1 *dTdp_o*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*(1 - upwinding)], [N_cell_connections, 2])
        
                    ! Derivative of Water Phase Flux Term W.R.T p
        dFdp_w = RESHAPE([-Thalf_w + c1 *dTdp_w*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*upwinding, &
            Thalf_w + c1 *dTdp_w*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*(1 - upwinding)], [N_cell_connections, 2])
!
        ! Derivative of Oil Phase Flux Term W.R.T s
        dFds_o = c1 * RESHAPE([dTds_o*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*upwinding, &
                  dTds_o*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*(1 - upwinding)], [N_cell_connections, 2])
        
                  ! Derivative of Water Phase Flux Term W.R.T s
        dFds_w = c1 * RESHAPE([dTds_w*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*upwinding, &
                  dTds_w*(P(cell_connections(:, 2)) - P(cell_connections(:, 1)))*(1 - upwinding)], [N_cell_connections, 2])
!
!
        !!!!! ====DERIVATIVE OF ACCUMILATION TERM==== !!!!!

        ! Derivative of Oil Phase Acc Term W.R.T p
        dA_o_coefdp = (dx*dy*dz/(rho_o_std*dt))*(poro*drhodp_o + rhoo*dporodp) 
        dA_odp = dA_o_coefdp*((1 - Sw)*(Co + Cr)*(P - P_holder(Prev_t_step, :))-(Sw-Sw_holder(Prev_t_step,:))) + &
                    A_o_coef*(1 - Sw)*(Co + Cr)
        
                    ! Derivative of Water Phase Acc Term W.R.T p
        dA_w_coefdp = (dx*dy*dz/(rho_w_std*dt))*(poro*drhodp_w + rhow*dporodp)
        dA_wdp = dA_o_coefdp*(Sw*(Cw + Cr)*(P - P_holder(Prev_t_step, :)) + (Sw - Sw_holder(Prev_t_step, :))) + &
                    A_w_coef*Sw*(Cw + Cr)

        ! Derivative of Oil Phase Acc Term W.R.T s
        dA_ods = A_o_coef*((-1)*(Co + Cr)*(P - P_holder(Prev_t_step, :)) - 1)
        ! Derivative of Water Phase Acc Term W.R.T s
        dA_wds = A_w_coef*((Cw + Cr)*(P - P_holder(Prev_t_step, :)) + 1)

    !    !!!!! ====DERIVATIVE OF SOURCE & SINK TERM==== !!!!!
!
        ! Derivative of Oil Phase SS Term W.R.T p
        dQ_odp = c1 * WI * (mobo + dmobodp) *Well_Loc(2, :)

        ! Derivative of Water Phase SS Term W.R.T p
        dQ_wdp = c1 * WI * (mobw + dmobwdp) *Well_Loc(2, :) + c3 * (Qinj*drhodp_w/rho_w_std)*Well_Loc(1, :)

        ! Derivative of Oil Phase SS Term W.R.T s
        dQ_ods = c1 *  WI * dmobods * (P - BHP) *Well_Loc(2, :)

        ! Derivative of Water Phase SS Term W.R.T s
        dQ_wds = c1 * WI * dmobwds * (P - BHP) *Well_Loc(2, :)

        DO i = 1, N_cell_connections

            c_1 = cell_connections(i, 1)
            c_2 = cell_connections(i, 2)
            !Locating Oil Phase Flux in Jacobian w.r.t p : 1/4 
            JAC(IDX_MAT(c_1, c_1))                 = JAC(IDX_MAT(c_1, c_1)) + dFdp_o(i, 1)
            JAC(IDX_MAT(c_1, c_2))                 =  dFdp_o(i, 2)
            JAC(IDX_MAT(c_2, c_1))                 = -dFdp_o(i, 1)
            JAC(IDX_MAT(c_2, c_2))                 = JAC(IDX_MAT(c_2, c_2)) - dFdp_o(i, 2)
            ! Locating Water Phase Flux in Jacobian w.r.t p : 2/4 
            JAC(IDX_MAT(c_1+N_cells, c_1))         = JAC(IDX_MAT(c_1+N_cells, c_1)) + dFdp_w(i, 1)
            JAC(IDX_MAT(c_1+N_cells, c_2))         =  dFdp_w(i, 2)
            JAC(IDX_MAT(c_2+N_cells, c_1))         = -dFdp_w(i, 1)
            JAC(IDX_MAT(c_2+N_cells, c_2))         = JAC(IDX_MAT(c_2+N_cells, c_2)) - dFdp_w(i, 2)
            !Locating Oil Phase Flux in Jacobian w.r.t s : 3/4 
            JAC(IDX_MAT(c_1, c_1+N_cells))         = JAC(IDX_MAT(c_1, c_1+N_cells)) + dFds_o(i, 1)
            JAC(IDX_MAT(c_1, c_2+N_cells))         =  dFds_o(i, 2)
            JAC(IDX_MAT(c_2, c_1+N_cells))         = -dFds_o(i, 1)
            JAC(IDX_MAT(c_2, c_2+N_cells))         = JAC(IDX_MAT(c_2, c_2+N_cells)) - dFds_o(i, 2)
            !Locating Water Phase Flux in Jacobian w.r.t s : 4/4  
            JAC(IDX_MAT(c_1+N_cells, c_1+N_cells)) = JAC(IDX_MAT(c_1+N_cells, c_1+N_cells)) + dFds_w(i, 1)
            JAC(IDX_MAT(c_1+N_cells, c_2+N_cells)) =  dFds_w(i, 2)
            JAC(IDX_MAT(c_2+N_cells, c_1+N_cells)) = -dFds_w(i, 1)
            JAC(IDX_MAT(c_2+N_cells, c_2+N_cells)) = JAC(IDX_MAT(c_2+N_cells, c_2+N_cells)) - dFds_w(i, 2)

        END DO

        DO i = 1, N_cells
            ! Oil Phase Full Jacobian w.r.t p : 1/4
            JAC(IDX_MAT(i, i))                 = JAC(IDX_MAT(i, i))                 - dA_odp(i) - dQ_odp(i)
            ! Water Phase Full Jacobian w.r.t p : 2/4 
            JAC(IDX_MAT(i+N_cells, i))         = JAC(IDX_MAT(i+N_cells, i))         - dA_wdp(i) - dQ_wdp(i)
            ! Oil Phase Full Jacobian w.r.t s : 3/4 
            JAC(IDX_MAT(i, i+N_cells))         = JAC(IDX_MAT(i, i+N_cells))         - dA_ods(i) - dQ_ods(i)
            ! Water Phase Full Jacobian w.r.t s : 4/4  
            JAC(IDX_MAT(i+N_cells, i+N_cells)) = JAC(IDX_MAT(i+N_cells, i+N_cells)) - dA_wds(i) - dQ_wds(i)
        END DO

    END SUBROUTINE get_residual_jacobian_triplet

END MODULE Residual_Jacobian_triplet