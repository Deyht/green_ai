program main
	implicit none
	integer :: omp_get_num_threads, omp_get_thread_num
	integer :: ind, nb_threads, i, j, k, dx, dy, dz, t1, t2, rate
	real (kind=8) :: acc
	integer, parameter :: v_size = 1024, rad = 3
	real, allocatable :: A(:,:,:), B(:,:,:)
	allocate(A(v_size,v_size,v_size))
	allocate(B(v_size,v_size,v_size))

	do i = 1, v_size
		do j = 1, v_size
			do k = 1,v_size
				A(k,j,i) = i+j+k
			end do
		end do
	end do
	
	call system_clock(t1, rate)

	!$OMP PARALLEL SHARED(A, B, nb_threads) private(ind, j, k, dx, dy, dz, acc)
	ind = omp_get_thread_num()
	if(ind == 0) then
		nb_threads = omp_get_num_threads()
		write (*,*) "There is currently	", nb_threads, "threads running"
	end if
	
	!$OMP DO SCHEDULE(DYNAMIC,1)
	do i = 1+rad, v_size-rad
		do j = 1+rad, v_size-rad
			do k = 1+rad,v_size-rad
				acc = 0.0
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
	!$OMP END DO
	!$OMP END PARALLEL

	call system_clock(t2, rate)
	write(*,*) B(v_size/2,v_size/2,v_size/2)
	write(*,*) "Compute time:", real(t2-t1)/rate, "sec"
end program main
