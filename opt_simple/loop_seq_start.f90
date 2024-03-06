
program main
	implicit none
	integer :: i, j, k, dx, dy, dz
	integer, parameter :: v_size = 512, rad = 3
	real, allocatable :: A(:,:,:), B(:,:,:)
	real (kind=8) :: acc
	allocate(A(v_size,v_size,v_size))
	allocate(B(v_size,v_size,v_size))

	do i = 1, v_size
		do j = 1, v_size
			do k = 1,v_size
				A(k,j,i) = i+j+k
			end do
		end do
	end do	

	B(:,:,:) = 0.0

	! Add your hown naive 6-loop implementation here
	

	write(*,*) B(v_size/2,v_size/2,v_size/2)
end program main
