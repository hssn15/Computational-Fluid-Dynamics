MODULE Properties

    IMPLICIT NONE
    SAVE
    PRIVATE
    PUBLIC :: porosity, rho_o, Kr_o, mu_o, rho_w, Kr_w, mu_w, DporosityDp, Drho_oDp, DKr_oDs, Dmu_oDp, Drho_wDp, DKr_wDs, Dmu_wDp


    CONTAINS
        !Pororsity 
        FUNCTION porosity(P, Cr, Pref, phi) RESULT(poro)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Cr, Pref, phi
            REAL(KIND=8), DIMENSION(size(P)) :: poro
            poro = phi * EXP(Cr * (P - Pref))
        END FUNCTION porosity
    
        ! Density of Oil
        FUNCTION rho_o(P, Co, Pref, rho_o_std) RESULT(rhoo)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Co, Pref, rho_o_std
            REAL(KIND=8), DIMENSION(size(P)) :: rhoo
            rhoo = rho_o_std * EXP(Co * (P - Pref))
        END FUNCTION rho_o
    
        ! Relative Permeability of Oil
        FUNCTION Kr_o(Sw, Somin, Swmin, Kro_) RESULT(kro)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: Sw
            REAL(KIND=8), INTENT(IN) :: Somin, Swmin, Kro_
            REAL(KIND=8), DIMENSION(size(Sw)) :: kro
            kro = Kro_ * ((1 - Sw - Somin) / (1 - Somin - Swmin))**3
        END FUNCTION Kr_o
    
        ! Viscosity of Oil
        FUNCTION mu_o(P, Co, Pref) RESULT(Viscosity)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Co, Pref
            REAL(KIND=8), DIMENSION(size(P)) :: Viscosity
            Viscosity = 1.0127 * EXP(-5 * Co * (P - Pref))
        END FUNCTION mu_o
    
        ! Density of Water
        FUNCTION rho_w(P, Cw, Pref, rho_w_std) RESULT(rhow)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Cw, Pref, rho_w_std
            REAL(KIND=8), DIMENSION(size(P)) :: rhow
            rhow = rho_w_std * EXP(Cw * (P - Pref))
        END FUNCTION rho_w
    
        ! Relative Permeability of Water
        FUNCTION Kr_w(Sw, Somin, Swmin, Krw_) RESULT(krw)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: Sw
            REAL(KIND=8), INTENT(IN) :: Somin, Swmin, Krw_
            REAL(KIND=8), DIMENSION(size(Sw)) :: krw
            krw = Krw_ * ((Sw - Swmin) / (1 - Somin - Swmin))**2
        END FUNCTION Kr_w
    
        ! Viscosity of Water
        FUNCTION mu_w(P, Cw, Pref) RESULT(Viscosity)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Cw, Pref
            REAL(KIND=8), DIMENSION(size(P)) :: Viscosity
            Viscosity = 1.0 * EXP(-Cw * (P - P)) ! Constant Water Viscosity = 1 cp
        END FUNCTION mu_w

        !=========================DERIVATIVES OF PTOPERTIES=========================!
        ! Derivative of Porosity with respect to p 
        FUNCTION DporosityDp(P, Cr, Pref, phi) RESULT(dporodp)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Cr, Pref, phi
            REAL(KIND=8), DIMENSION(size(P)) :: dporodp
            dporodp = Cr * phi * EXP(Cr * (P - Pref))
        END FUNCTION DporosityDp
    
        ! Derivative of Density of Oil with respect to p
        FUNCTION Drho_oDp(P, Co, Pref, rho_o_std) RESULT(drhoodp)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Co, Pref, rho_o_std
            REAL(KIND=8), DIMENSION(size(P)) :: drhoodp
            drhoodp = Co *rho_o_std * EXP(Co * (P - Pref))
        END FUNCTION Drho_oDp
    
        ! Derivative of Relative Permeability of Oil with respect to S
        FUNCTION DKr_oDs(Sw, Somin, Swmin, Kro_) RESULT(dkrods)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: Sw
            REAL(KIND=8), INTENT(IN) :: Somin, Swmin, Kro_
            REAL(KIND=8), DIMENSION(size(Sw)) :: dkrods
            dkrods = -3 * Kro_ * ((1 - Sw - Somin)**2 / (1 - Somin - Swmin)**3)
        END FUNCTION DKr_oDs

        ! Derivative of Viscosity of Oil with respect to p
        FUNCTION Dmu_oDp(P, Co, Pref) RESULT(dViscositydp)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Co, Pref
            REAL(KIND=8), DIMENSION(size(P)) :: dViscositydp
            dViscositydp = -5 * Co * 1.0127 * EXP(-5 * Co * (P - Pref))
        END FUNCTION Dmu_oDp
        
        ! Derivative of Density of Water with respect to p
        FUNCTION Drho_wDp(P, Cw, Pref, rho_w_std) RESULT(drhowdp)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Cw, Pref, rho_w_std
            REAL(KIND=8), DIMENSION(size(P)) :: drhowdp
            drhowdp = Cw * rho_w_std * EXP(Cw * (P - Pref))
        END FUNCTION Drho_wDp
    
        ! Derivative of Relative Permeability of Waterwith respect to S
        FUNCTION DKr_wDs(Sw, Somin, Swmin, Krw_) RESULT(dkrwds)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: Sw
            REAL(KIND=8), INTENT(IN) :: Somin, Swmin, Krw_
            REAL(KIND=8), DIMENSION(size(Sw)) :: dkrwds
            dkrwds = 2 * Krw_ * ((Sw - Swmin) / (1 - Somin - Swmin)**2)
        END FUNCTION DKr_wDs
    
        ! Derivative of Viscosity of Water with respect to p
        FUNCTION Dmu_wDp(P, Cw, Pref) RESULT(dViscositydp)
            IMPLICIT NONE
            REAL(KIND=8), DIMENSION(:), INTENT(IN) :: P
            REAL(KIND=8), INTENT(IN) :: Cw, Pref
            REAL(KIND=8), DIMENSION(size(P)) :: dViscositydp
            dViscositydp = -Cw * 1.0 * EXP(-Cw * (P - P)) ! Derv of Constant Water Viscosity = 0
        END FUNCTION Dmu_wDp

END MODULE Properties