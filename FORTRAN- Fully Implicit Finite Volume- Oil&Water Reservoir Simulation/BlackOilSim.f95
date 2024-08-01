INCLUDE "linear_solvers.f95"
!INCLUDE "residual.f95"
INCLUDE "invMat.f95"
INCLUDE "residual_triplet_JAC.f95"
INCLUDE "gmres.f95"

PROGRAM MAIN

    USE Nonsymmetric_Methods
    USE dictionary_module
    !USE Residual_Jacobian ! Returns residual vector and jacobian matrix of O and W phases
    USE Residual_Jacobian_triplet 
    USE LinPack  ! Linear Solver for AX = B (JR * dX = R)
    USE Transmissibility ! Return Geometric Transmissibility and Transmissibility between Cells for O and W phases
    USE InverseMat
    IMPLICIT NONE

    TYPE(dictionary_type) :: Property_Dict

    !================================================ DEFINING INPUTS AND OUTPUTS ================================================!

    character(len=20) :: filenameP, filenameS, filename1, filename2, filename3
    REAL(KIND=8) :: start, finish, start1, finish1
    REAL(KIND=8),    ALLOCATABLE :: K(:)
    REAL(KIND=8), ALLOCATABLE :: cell_connections_real(:, :)
    INTEGER(KIND=4), ALLOCATABLE :: cell_connections(:, :)
    INTEGER(KIND=4):: N_cell_connections = 420 ! total cell connections
    INTEGER(KIND=4):: N_cells = 225            ! total number of cells
    REAL(KIND=8) :: dx, dy, dz                ! cell dimensions
    REAL(KIND=8) :: rho_o_std, rho_w_std, Cr, Co, Cw, Pref, phi, Swmin, Somin, Kro_, Krw_ ! Oil and Water phase variables
    REAL(KIND=8), PARAMETER :: c1 = 6.2283E-3, c2 = 1.0, c3 = 5.5146 ! Calibration Coefficients
    ! Well Parameters
    REAL(KIND=8) :: BHP, Qinj, rw, re, S, WI ! Bottom Hole Pressure, Water Inj Rate, WellBore rad, equivalent rad, Skin, Well Idx
    REAL(KIND=8) :: well_1, well_2           ! Injector Loc, Producer Loc
    REAL(KIND=8), ALLOCATABLE :: Well_Loc(:, :) ! Well Map
    ! Variables
    REAL(KIND=8), ALLOCATABLE :: X(:) ! To Hold Current Time Step
    REAL(KIND=8), ALLOCATABLE :: Sw_holder(:, :), P_holder(:, :) ! To Hold Past Time Steps
    ! Results
    REAL(KIND=8), ALLOCATABLE :: TGeo(:)  ! CalculateD Geometric Transmissibility beforehand as it's const
    REAL(KIND=8), ALLOCATABLE :: RES(:)   !Residual Vector
    REAL(KIND=8), ALLOCATABLE :: JAC(:) !Jacobian Triplet
    REAL(KIND=8), PARAMETER :: TOL = 1.0D-6 ! Newton tolerance for mass balance
    REAL(KIND=8) :: MB_ERROR
    INTEGER(KIND=4) :: i , j, errorflag
    ! Time Steps
    REAL(KIND=8) :: Days = 400 ! Sim ualtion Timeline
    REAL(KIND=8) :: dt = 1, CPU_TIME_JAC_RES = 0, CPU_TIME_SOLVER = 0    ! Time step : 1 Day
    REAL(KIND=8), DIMENSION(400):: RESIDUAL_ERROR
    INTEGER(KIND=4) :: N_steps, Prev_t_step, iter ! Tot Num of Steps, store previous t-step

    !=============================================================================================================================!

    !========================= SOLVER INPUT SET ===========================!
    INTEGER(KIND=4)              :: order_l_sys , nz_num                   !
    REAL(KIND=4), ALLOCATABLE :: ia(:) , ja(:)                             !
    REAL(KIND=4), ALLOCATABLE :: IDX_MAT(:, :)                             !
    INTEGER(KIND=4)              :: itr_max = 100, mr = 100                !
    REAL(KIND=8)                 :: tol_abs = 1E-4, tol_rel = 1E-4         !
    REAL(KIND=8),    ALLOCATABLE :: PartialX(:)                            !
                                                                           !
    order_l_sys = 2*N_cells                                                !
    nz_num = (2*N_cell_connections + N_cells)*4                            !
                                                                           !
    !======================================================================!

    Prev_t_step = 1 
    N_steps = INT(1 + Days/dt)

    !=================================== ALLOCATING =================================!
    ALLOCATE(ia(nz_num), ja(nz_num))
    ALLOCATE(IDX_MAT(N_cells*2, N_cells*2))
    ALLOCATE(PartialX(N_cells*2))   

    ALLOCATE(P_holder(N_steps, N_cells), Sw_holder(N_steps, N_cells))
    ALLOCATE(X(N_cells*2))
    ALLOCATE(K(N_cells))
    ALLOCATE(cell_connections_real(N_cell_connections, 2))
    ALLOCATE(cell_connections(N_cell_connections, 2))
    ALLOCATE(TGeo(N_cell_connections))
    ALLOCATE(RES(N_cells*2))
    ALLOCATE(JAC(nz_num))
    ALLOCATE(Well_Loc(2, N_cells))
    !================================================================================!

    !============================= TRIPLET MATRIX DATA ==============================!
    OPEN (14, file = 'rows.dat', status = 'old')
    DO i = 1,nz_num  
       READ(14,*) ia(i)
    END DO 
    CLOSE(14) 

    OPEN (15, file = 'cols.dat', status = 'old')
    DO i = 1,nz_num  
       READ(15,*) ja(i)
    END DO 
    CLOSE(15)

    OPEN (16, file = 'connection_indexing.dat', status = 'old')
    DO i = 1,N_cells*2  
       READ(16,*) IDX_MAT(i, :)
    END DO 
    CLOSE(16)
    !================================================================================!

    !!!!!!!!======= ASSIGNING INPUTS AND OUTPUTS  WITH VALUES=======!!!!!!!!

    ! System Properties

    rho_o_std = 45.0  ! #lb/ft^3
    rho_w_std = 62.4  ! #lb/ft^3
    Cr        = 3e-6  ! Rock Compressibility
    Co        = 1e-5  ! Oil Compressibility
    Cw        = 3e-6  ! Water Compressibility
    Pref      = 2800  ! Reference Pressure
    phi       = 0.2   ! Initial Porosity
    Swmin     = 0.1   ! Reduced Water Saturation
    Somin     = 0.2   ! Reduced Oil Saturation
    Kro_      = 0.7
    Krw_      = 0.08

    ! Permeability Distribution
    OPEN (12, file = 'permx.dat', status = 'old')
    DO i = 1,N_cells  
       READ(12,*) K(i)
    END DO 
    CLOSE(12)

    ! Model Properties
    dx = 30 ! Cell Dimensions
    dy = 30 
    dz = 30 

    OPEN (13, file = 'edge_array.dat', status = 'old')
    DO i = 1,N_cell_connections  
       READ(13,*) cell_connections_real(i, :)
    END DO 
    CLOSE(13)
    cell_connections = INT(cell_connections_real+1)
    ! Well Properties
    BHP  = 2900
    Qinj = 300
    rw   = 0.35
    re   = 0.14*sqrt(dx**2 + dy**2)
    S    = 0 ! Skin Term
    well_1 = 1  ! Injector
    well_2 = 225 ! Producer
    Well_Loc = 0
    Well_Loc(1, INT(well_1)) = -1
    Well_Loc(2, INT(well_2)) =  1
    WI = 2 * 3.141592653589793 * K(INT(well_2))* dz / (log(re/rw) + S) ! Well Index


    ! Adding to the dictionary   
    CALL add_to_dictionary(Property_Dict, "c1",    c1   )
    CALL add_to_dictionary(Property_Dict, "c2",    c2   )
    CALL add_to_dictionary(Property_Dict, "c3",    c3   ) 
    CALL add_to_dictionary(Property_Dict, "Cr",    Cr   )
    CALL add_to_dictionary(Property_Dict, "Co",    Co   )
    CALL add_to_dictionary(Property_Dict, "Cw",    Cw   )
    CALL add_to_dictionary(Property_Dict, "Pref",  Pref )
    CALL add_to_dictionary(Property_Dict, "phi",   phi  )
    CALL add_to_dictionary(Property_Dict, "Swmin", Swmin)
    CALL add_to_dictionary(Property_Dict, "Somin", Somin)
    CALL add_to_dictionary(Property_Dict, "Kro_",  Kro_ )
    CALL add_to_dictionary(Property_Dict, "Krw_",  Krw_ )
    CALL add_to_dictionary(Property_Dict, "dx",    dx   )
    CALL add_to_dictionary(Property_Dict, "dy",    dy   )
    CALL add_to_dictionary(Property_Dict, "dz",    dz   )
    CALL add_to_dictionary(Property_Dict, "dt",    dt   )
    CALL add_to_dictionary(Property_Dict, "BHP",   BHP  )
    CALL add_to_dictionary(Property_Dict, "Qinj",  Qinj )
    CALL add_to_dictionary(Property_Dict, "rw",    rw   )
    CALL add_to_dictionary(Property_Dict, "re",    re   )
    CALL add_to_dictionary(Property_Dict, "S",     S    )
    CALL add_to_dictionary(Property_Dict, "WI",    WI   )
    CALL add_to_dictionary(Property_Dict, "well_1", well_1)
    CALL add_to_dictionary(Property_Dict, "well_2", well_2)
    CALL add_to_dictionary(Property_Dict, "rho_o_std", rho_o_std)
    CALL add_to_dictionary(Property_Dict, "rho_w_std", rho_w_std)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! Adding First Time Step into Holders
    P_holder(1, :)  = 3000.0  ! Initial Pressure    
    Sw_holder(1, :) = 0.2     ! Initial Water Saturation

    DO i = 1, 225
        X(i) = 3000.0
    END DO

    DO i = 226, 450
        X(i) = 0.2
    END DO

    TGeo = T_Geo(K, dx, dy, dz, cell_connections, N_cell_connections)

    !===============================================================================================!
    !======================================= SOLVER : MGMRES =======================================!
    !===============================================================================================!
    PartialX = X*1D-02
    CPU_TIME_JAC_RES = 0
    CPU_TIME_SOLVER = 0
    DO Prev_t_step = 1, N_steps-1
        iter = 0
        MB_ERROR = 1.0D+3
        DO WHILE (MB_ERROR > TOL)
    
            CALL cpu_time(start)
            CALL get_residual_jacobian_triplet(X, P_holder, Sw_holder, Tgeo, K, cell_connections, N_cell_connections,&
                                        N_cells, N_steps, Property_Dict, Well_Loc, Prev_t_step, JAC, RES, INT(IDX_MAT))
            CALL cpu_time(finish)
            CPU_TIME_JAC_RES = CPU_TIME_JAC_RES + (finish - start)
    
            !PRINT *, "RES+JAC CALCULATION TIME = ", finish-start
            CALL cpu_time(start1)
            CALL mgmres_st(order_l_sys, nz_num, INT(ia), INT(ja), JAC, PartialX, RES, itr_max, mr, tol_abs, tol_rel)
            X = X - PartialX
            CALL cpu_time(finish1)
            CPU_TIME_SOLVER = CPU_TIME_SOLVER + (finish1 - start1)
            MB_ERROR = SQRT(DOT_PRODUCT(RES,RES))
    
            iter = iter + 1
    
        END DO
    
        RESIDUAL_ERROR(Prev_t_step) = SUM((RES**2)**0.5)/(2*N_cells)
        PRINT *, "T = ", Prev_t_step, 'iter = ', iter, "||  RESIDUAL ERROR = ", MB_ERROR/SIZE(RES)
    
        P_holder(Prev_t_step+1, :) = X(: N_cells)     
        Sw_holder(Prev_t_step+1, :) = X(N_cells+1 :)
    
    END DO
    
    PRINT *, "CPU_TIME_JAC_RES = ", CPU_TIME_JAC_RES
    PRINT *, "CPU_TIME_SOLVER  = ", CPU_TIME_SOLVER

    !============================================================================================!
    !====================================== SOLVER : DIREC ======================================!
    !============================================================================================!
    !DO Prev_t_step = 1, N_steps-1
    !
    !    iter = 0
    !    error = 1000
    !
    !    DO 
    !        IF (iter > 3) EXIT
    !        CALL cpu_time(start)
    !        CALL get_residual_jacobian(X, P_holder, Sw_holder, Tgeo, K, cell_connections, N_cell_connections,&
    !                                N_cells, N_steps, Property_Dict, Well_Loc, Prev_t_step, JAC, RES)
    !        CALL cpu_time(finish)
    !        PRINT *, "RES+JAC CALCULATION TIME = ", finish-start
    !        
    !        invJAC = FINDInv(JAC, N_cells*2, errorflag)
    !        
    !        X = X - MATMUL(invJAC, RES)
    !        error = SUM((RES**2)**0.5)/(2*N_cells)
    !        
    !        iter = iter + 1
    !
    !    END DO
    !    
    !    PRINT *, "T = ", Prev_t_step
    !    PRINT *, "  ||  error =  ",error
    !
    !    P_holder(Prev_t_step+1, :) = X(: N_cells)     
    !    Sw_holder(Prev_t_step+1, :) = X(N_cells+1 :)
    !
    !END DO

    !===============================================================================================!
    !====================================== SOLVER : GE PIVOT ======================================!
    !===============================================================================================!
    !DO Prev_t_step = 1, N_steps-1
    !    iter = 0
    !    DO 
    !        IF (iter > 3) EXIT
    !
    !        CALL cpu_time(start)
    !        CALL get_residual_jacobian(X, P_holder, Sw_holder, Tgeo, K, cell_connections, N_cell_connections,&
    !                                N_cells, N_steps, Property_Dict, Well_Loc, Prev_t_step, JAC, RES)
    !        CALL cpu_time(finish)
    !        CPU_TIME_JAC_RES = CPU_TIME_JAC_RES + (finish - start)
    !
    !        !PRINT *, "RES+JAC CALCULATION TIME = ", finish-start
    !        CALL cpu_time(start1)
    !        CALL GE_Pivot(JAC, RES, PartialX)
    !        X = X - PartialX
    !        CALL cpu_time(finish1)
    !        CPU_TIME_SOLVER = CPU_TIME_SOLVER + (finish1 - start1)
    !
    !        iter = iter + 1
    !
    !    END DO
    !
    !    RESIDUAL_ERROR(Prev_t_step) = SUM((RES**2)**0.5)/(2*N_cells)
    !    PRINT *, "T = ", Prev_t_step, "||  RESIDUAL ERROR = ",SUM((RES**2)**0.5)/(2*N_cells)
    !
    !    P_holder(Prev_t_step+1, :) = X(: N_cells)     
    !    Sw_holder(Prev_t_step+1, :) = X(N_cells+1 :)
    !
    !END DO
    !
    !PRINT *, "CPU_TIME_JAC_RES = ", CPU_TIME_JAC_RES
    !PRINT *, "CPU_TIME_SOLVER  = ", CPU_TIME_SOLVER

    !====================================================================================================!
    !====================================== SAVING DATA INTO FILES ======================================!
    !====================================================================================================!

    !filename3 = 'residual_error.txt'

    !OPEN(unit=12, file=filename3, status='new')

    !DO j = 1, 400
    !    WRITE(12, '(F10.3)', advance='no'), RESIDUAL_ERROR(j)
    !    IF (j < 400) THEN
    !        WRITE(12, '(A)', advance='no'), ' '  
    !    END IF
    !END DO
    !WRITE(12, *)  
    !CLOSE(12)
    

    filenameP = 'array_outputP.txt'
    filenameS = 'array_outputS.txt'
    OPEN(unit=10, file=filenameP, status='unknown')
    
    DO i = 1, N_steps
        DO j = 1, N_cells
            WRITE(10, '(F10.2)', advance='no'), P_holder(i, j)
            IF (j < N_cells) THEN
                WRITE(10, '(A)', advance='no'), ' '  
            END IF
        END DO
        WRITE(10, *)  
    END DO
    CLOSE(10)
    OPEN(unit=11, file=filenameS, status='unknown')
    
    DO i = 1, N_steps
        DO j = 1, N_cells
            WRITE(11, '(F10.2)', advance='no'), Sw_holder(i, j)
            IF (j < N_cells) THEN
                WRITE(11, '(A)', advance='no'), ' '  
            END IF
        END DO
        WRITE(11, *)  
    END DO
    CLOSE(11)

    DEALLOCATE(P_holder, Sw_holder)
    DEALLOCATE(X)
    DEALLOCATE(K)
    DEALLOCATE(cell_connections)
    DEALLOCATE(cell_connections_real)
    DEALLOCATE(TGeo)
    DEALLOCATE(RES)
    DEALLOCATE(JAC)
    DEALLOCATE(Well_Loc)
    DEALLOCATE(ia, ja)
    DEALLOCATE(PartialX)   

END PROGRAM MAIN