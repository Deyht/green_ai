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

	do i = 1+rad, v_size-rad
		do j = 1+rad, v_size-rad
			do k = 1+rad,v_size-rad
				acc = 0
				do dx = i-rad, i+rad
					do dy = j-rad, j+rad
						do dz = k-rad, k+rad
							acc = acc + A(dz, dy, dx)
						end do
					end do				
				end do
				B(k,j,i) = acc			
			end do
		end do
	end do
	

	write(*,*) B(v_size/2,v_size/2,v_size/2)
end program main
