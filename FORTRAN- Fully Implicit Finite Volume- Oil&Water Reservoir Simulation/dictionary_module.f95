 ! ||||||||||||||||||||||||||||||||||||||DICTIONARY MODULE|||||||||||||||||||||||||||||||||||||| !
MODULE dictionary_module
    IMPLICIT NONE
    SAVE
    PRIVATE
    PUBLIC :: dictionary_type, add_to_dictionary, get_from_dictionary

    INTEGER(KIND=4), parameter :: max_size = 100

    TYPE :: dictionary_type
        CHARACTER(len=32) :: keys(max_size)
        REAL(KIND=8) :: values(max_size)
        INTEGER(KIND=4) :: count = 0
    END TYPE dictionary_type

    CONTAINS

    subroutine add_to_dictionary(dict, key, value)
        TYPE(dictionary_type), intent(inout) :: dict
        CHARACTER(len=*), INTENT(IN) :: key
        REAL(KIND=8), INTENT(IN) :: value

        IF (dict%count < max_size) then
            dict%count = dict%count + 1
            dict%keys(dict%count) = key
            dict%values(dict%count) = value
        ELSE
            PRINT *, "Dictionary is full!"
        END IF 
    END subroutine add_to_dictionary

    REAL(KIND=8) FUNCTION get_from_dictionary(dict, key)
        TYPE(dictionary_type), INTENT(IN) :: dict
        CHARACTER(len=*), INTENT(IN) :: key
        INTEGER(KIND=4) :: i

        get_from_dictionary = -1.0  ! Default value if key not found
        DO i = 1, dict%count
            IF (trim(dict%keys(i)) == trim(key)) THEN
                get_from_dictionary = dict%values(i)
                RETURN
            END IF 
        END DO
    END FUNCTION get_from_dictionary

END MODULE dictionary_module
 ! ||||||||||||||||||||||||||||||||||||||DICTIONARY MODULE|||||||||||||||||||||||||||||||||||||| !