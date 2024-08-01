MODULE Transmissibility
    IMPLICIT NONE
    SAVE
    PRIVATE
    PUBLIC :: T_Geo, T_full, dTdp_dTds, up

    CONTAINS 
    
        ! Geometric Transmissibility
        FUNCTION T_Geo(K, dx, dy, dz, cell_connections, N_cell_connections) RESULT(TGeo)
            IMPLICIT NONE
            INTEGER(KIND=4), INTENT(IN) :: N_cell_connections
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: K ! Permeability
            INTEGER(KIND=4), DIMENSION(N_cell_connections, 2), INTENT(IN) :: cell_connections
            REAL(KIND=8), INTENT(IN) :: dx, dy, dz
            REAL(KIND=8), DIMENSION(N_cell_connections) :: TGeo
            
            TGeo = (dz*dy/dx) * 2 / (1/K(cell_connections(:, 1)) + 1/K(cell_connections(:, 2)))
        END FUNCTION T_Geo
    
        FUNCTION T_full(kr, mu, rho, rho_std, Tgeo, cell_connections, N_cell_connections, N_cells, a) RESULT(Thalf)
            IMPLICIT NONE
            INTEGER(KIND=4), INTENT(IN) :: N_cell_connections, N_cells
            REAL(KIND=8), DIMENSION(N_cells), INTENT(IN) :: kr, mu, rho
            REAL(KIND=8), INTENT(IN) :: rho_std
            REAL(KIND=8), DIMENSION(N_cell_connections), INTENT(IN) :: Tgeo
            INTEGER(KIND=4), DIMENSION(N_cell_connections, 2), INTENT(IN) :: cell_connections
            INTEGER(KIND=4), DIMENSION(N_cell_connections), INTENT(IN) :: a
            REAL(KIND=8), DIMENSION(N_cells) :: mobility
            REAL(KIND=8), DIMENSION(N_cell_connections) :: Thalf
    
            mobility = kr*rho/(mu*rho_std)
            Thalf = Tgeo*(mobility(cell_connections(:, 1)) * a + mobility(cell_connections(:, 2)) * (1 - a))
        END FUNCTION T_full
    
        SUBROUTINE dTdp_dTds(kr, mu, rho, rho_std, dkrds, dmudp, drhodp, Tgeo, cell_connections, N_cell_connections, N_cells, a, &
                                                                                                                        dTdp, dTds)
            IMPLICIT NONE
            INTEGER(KIND=4), INTENT(IN) :: N_cell_connections, N_cells
            REAL(KIND=8), DIMENSION(N_cells), INTENT(IN) :: kr, mu, rho, dkrds, dmudp, drhodp
            REAL(KIND=8), INTENT(IN) :: rho_std
            REAL(KIND=8), DIMENSION(N_cell_connections), INTENT(IN) :: Tgeo
            INTEGER(KIND=4), DIMENSION(N_cell_connections, 2), INTENT(IN) :: cell_connections
            INTEGER(KIND=4), DIMENSION(N_cell_connections), INTENT(IN) :: a
            REAL(KIND=8), DIMENSION(N_cells) :: dmobdp, dmobds
            REAL(KIND=8), DIMENSION(N_cell_connections), INTENT(OUT):: dTdp, dTds
            ! Derivative of Mobility w.r.t p and s
            dmobdp = (kr/rho_std)*(rho*(-1/mu**2)*dmudp  + (1/mu)*drhodp)
            dmobds = (dkrds*rho/(mu*rho_std))
            ! Derivative of Transmissibility w.r.t p and s
            dTdp = Tgeo*(dmobdp(cell_connections(:, 1)) * a + dmobdp(cell_connections(:, 2)) * (1 - a)) 
            dTds = Tgeo*(dmobds(cell_connections(:, 1)) * a + dmobds(cell_connections(:, 2)) * (1 - a))

        END SUBROUTINE dTdp_dTds


    ! Upwinding
        FUNCTION up(P, cell_connections, N_cell_connections) RESULT(a)
            IMPLICIT NONE
            INTEGER(KIND=4), INTENT(IN) :: N_cell_connections
            INTEGER(KIND=4), DIMENSION(N_cell_connections, 2), INTENT(IN) :: cell_connections
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            INTEGER(KIND=4), DIMENSION(N_cell_connections) :: a
            a = 0
            WHERE(P(cell_connections(:, 1)) > P(cell_connections(:, 2))) a = 1
        END FUNCTION up
    
END MODULE Transmissibility