!!!**********************************************!!!
!                                                  !
!             声爆/气动快速预测系统                !
!                    FABOOM V1                     !
!                    2020.6.5                      ! 
!                     丁玉临                       !
!                                                  ! 
!!!**********************************************!!!  

!!!**********************************************************************!!!
!                                                                          !
!                               修改日志                                   !
!2020.9.1  添加panair压力系数分布和剖面信息输入与输出（A502in和A502out）   !
!2020.9.2修改了输出机翼剖面信息的bug                                       ! 
!2020.11.27修改panair out文件读取格式化读入数据                            !
!2020.12.3修改体积截面输出坐标相减bug                                      !
!2021.7.14加入计时功能                                                     ! 
!2021.11.24 添加Burgers-Hayes方法来确定激波位置                            !
!2021.11.25 修改体积等效截面积截取bug                                      !
!                                                                          ! 
!!!**********************************************************************!!! 

    program FABOOM
    implicit none
    integer T1,T2
    integer KG,KG1
    character*256 str
   
    !=======================计时=======================！
    call SYSTEM_CLOCK(T1)
    open(unit=1,file="./indata/FABoom.in")
    do while(.TRUE.)
        read(1,'(A)')str
        if (index(str,"volume or lift")/= 0)then
        exit
        end if
    end do
    read(1,*)KG,KG1
    close(1)
    if(KG1==1)then
     write(*,*)"   "
    Write(*,*)"================================================================"
    Write(*,*)"=                                                              ="
    write(*,*)"=     000000     0           00000   0000   0000  0     0      ="
    write(*,*)"=     0         0 0          0    0 0    0 0    0 00   00      ="
    Write(*,*)"=     000000   0   0  00000  00000  0    0 0    0 0 0 0 0      ="
    Write(*,*)"=     0       0000000        0    0 0    0 0    0 0  0  0      ="
    Write(*,*)"=     0      0       0       00000   0000   0000  0     0      ="
    Write(*,*)"=                                                              ="
    Write(*,*)"================================================================"
    else
        Write(*,*)"执行FA-BOOM"
    end if 
    
   
    
    !===========生成A502程序的输入文件A502.in==========！
    if(KG1==1)then
    call A502in
    end if 
    
    if(KG1==2)then
        
    call A502in_2
    
    end if 
    
    if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>----------- 完成A502输入程序生成 -----------<<<"
    end if 
   
    !===========执行A502计算程序==========！
    if(KG==2)then
    if(KG1==1)then    
    write(*,*)"           "
    write(*,*)">>>--------------- 执行A502计算 ---------------<<<"
    end if 
    call SYSTEM('cd A502 && A502.exe')
    if(KG1==1)then 
    write(*,*)"           "
    write(*,*)">>>--------------- 完成A502计算 ---------------<<<"
    end if 
    
    !===========删除一些叼毛文件==========！
    call system('cd A502 && del ft*')
    call system('cd A502 && del rwms*')
    call system('cd A502 && del fort*')
    call system('cd A502 && del news*')
    call system('cd A502 && del nlians*')
    call system('cd A502 && del nlirhs*')
    if(KG1==1)then 
    write(*,*)"           "
    write(*,*)">>>--------- 完成A502无用输出文件删除 ---------<<<"
    end if
    !===========执行A502输出文件处理与升力等效截面积分布处理==========！
    if(KG1==1)then
    call A502out1 
    write(*,*)"           "
    write(*,*)">>>- 完成A502输出文件处理与升力等效截面积计算 -<<<"  
    else
        call A502out2 
    end if
    end if
    !=====================执行体积等效截面积计算======================！
    if(Kg/=3)then
    call slice
    if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>---------- 完成体积等效截面积计算 ----------<<<" 
     end if
    !===========================截面积计算============================！
    if(KG==1)then
    call area_distribution1
    if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>------------ 完成等效截面积计算 ------------<<<" 
    end if
    end if
    
    if(KG==2)then
    call area_distribution2
     if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>------------ 完成等效截面积计算 ------------<<<" 
     end if
     end if
    end if
    
    !=====================执行等效截面积2阶导数计算===================！
    if(KG1==1.or.KG1==2.or.KG==3)then
    !if(KG1==1.or.KG1==2.AND.kg==2.or.KG==3)then
    call diferrential
    if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>---------- 完成等效截面积2阶导计算 ---------<<<"
    end if
    !==========================执行F函数计算==========================！
    if(KG==3.AND.KG1==2)then
         call F_function2
         else
    call F_function
    if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>--------- 完成F函数和未修正波形计算 --------<<<"
    end if
    !========================执行激波修正计算=========================！
    !call shock_correct
    call shock_correct_BH
    if(KG1==1)then
    write(*,*)"           "
    write(*,*)">>>------ 完成波形修正计算和近场波形输出 ------<<<"
    end if
    end if
    !=======================计时=======================！
    call SYSTEM_CLOCK(T2)
    if(KG1==1)then
     write(*,'(A,F10.4,A)')"耗时为",(T2-T1)/10000.0,"s"
     end if
     end if
    if(KG1==1)then 
    Write(*,*)"======================================================================="
    Write(*,*)"=                                                                     ="
    Write(*,*)"=      0000000        00000000          0                000000       ="
    write(*,*)"=         0            0000 0       00  0          0      0000        ="
    write(*,*)"=        000           0  0 0      00 000000        0     0000        ="
    Write(*,*)"=    00000000000       0000 0     0     0         000   00000000      ="
    Write(*,*)"=       00 00      000000000000   0000000000000     0   0  00  0      ="
    Write(*,*)"=      0 0 0 0         0000 0           0           0   00000000      ="
    Write(*,*)"=     0  0 0  0        0  0 0           0           0   0  00  0      ="
    Write(*,*)"=        0 0           0000 0           0         000   00000000      ="
    Write(*,*)"=       0  0               00           0           000000000000000   ="
    Write(*,*)"=                                                                     ="
    Write(*,*)"======================================================================="
     end if
     
    end program FABOOM
    
    
    
    
    
    
    !===================================================================================================！
    !                                                                                                   ！
    !                                读取输入文件生成A502.in文件的子程序FABOOM                          ！
    !                                读取输入文件生成A502.in文件的子程序FABOOM                          ！
    !                                读取输入文件生成A502.in文件的子程序FABOOM                          ！
    !                                     2020.7.9更新几何外形dat文件输出                               ！
    !===================================================================================================！
    
    
    !program FABOOM
    subroutine A502in
    implicit none
    
    character(len=20)::filename,str
    integer i,j,k,ii,jj,kk
    integer Nmesh,Np,Nrow,Nleft,Numcut,meshcut
    integer Nkt1,Nkt5,Nkt18,Nkt181,Nkt20
    integer Narea,Nnear
    integer,allocatable::Okt1(:),Okt5(:),Okt18(:),Okt181(:),Okt20(:),Ekt18(:),Ekt181(:),Ordercut(:)
    real temp,mach,AOA,H,rou,p,R,PHI,checkface
    real(kind=8) L,Lm,S
    real Vtx
    real cutlength,cutratio
    real x0,x1,xs,xe
    real::pai=3.141592658
    integer,allocatable::Nij(:,:)
    real,allocatable::x(:),y(:),z(:),ycut(:)
    real,allocatable::near(:,:)
    !write(*,*)'输入文件名'
    !read(*,*)filename
    !write(*,*)filename
    open(unit=1,file="indata/geo.x",action='read',status='old')            !!!!2020.6.19修改输入文件至indata
    open(unit=2,file="A502/A502.in",action="write",status="replace")
    open(unit=3,file="indata/FABoom.in",action='read',status='old')
    open(unit=4,file="indata/geo_information1.in",action='read',status='old')
    !open(unit=5,file="indata/atmospheric_profile.in",action='read',status='old')
    open(unit=6,file="tempdata/geometry.dat",action='write',status='replace')
    
    !================================!!子程序A502in变量说明!!!==================================！
    !Nmesh,Np,Nrow,Nleft               网格块数，当前网格的网格点数，数据块的行数，数据最后一行的数据个数
    !Nkt1,Nkt5,Nkt18,Nkt181,Nkt20      kt1边界网格数，kt5边界网格数，kt18边界网格数，kt18.1边界网格数，kt20边界网格数
    !Narea,Nnear                       面积分布中所用的点数，近场信号提取的点数
    !Okt1,Okt5,Okt18,Okt181,Okt20      kt1网格编号，kt5网格编号，kt18网格编号，kt18.1网格编号，kt20网格编号
    !Ekt18,Ekt181                      kt18网格搭接尾迹的edge编号，kt18.1网格搭接尾迹的edge编号     
    !temp,mach,AOA,H,rou,p,R,PHI       活动变量，马赫数，迎角，飞行高度，密度，无穷来流静压，提取位置的体长倍数，周向角
    !L,Lm,S                            飞机长度，计算模型长度，半模参考面积
    !Vtx                               尾迹延伸末尾的x坐标
    !x0,x1,xs,xe                       飞机头部x坐标，飞机尾部x坐标，近场提取点开始坐标，近场提取点结束坐标
    !Nij,x,y,z,near                    网格维数数组，x坐标，y坐标，z坐标，近场提取点坐标数组
    !checkface                         检查是物面网格的标记
    !================================!!子程序A502in变量说明!!!==================================！
    
    
    
    !========================================开始读取FABOOM，写A502in文件开头
    read(3,*)str,str,str
    write(2,1)"$TITLE ",str
    read(3,*)str
    write(2,'(A)')str,"Created by PANIN","$DATACHECK","0.","$symmetry - xz plane"
    write(2,'(2(A10))')"1.","0."
    write(2,'(A)')"$mach number"
    read(3,*)
    read(3,*)mach,AOA,H,rou,p,R,PHI
    write(2,'(F10.3)')mach
    !write(2,'(A)')"$cases - no. of solutions","1.","$angles-of-attack","0."
    write(2,'(A)')"$cases - no. of solutions","1.","$angles-of-attack"
    write(2,'(F10.5)')AOA
    write(2,'(F10.5)')AOA
    write(2,'(A)')"$printout options","0.        0.        0.        1.        0.        0.",".0        0.        0.        0.        0."
    write(2,'(A)')"$references for accumulated forces and moments","0.        0.        0."
    read(3,*)
    read(3,*)L,Lm,temp,S
    !write(str,*)S
    write(2,'(F10.6,3(A10))')S,"        1.","        1.","        0."    
         
    !========================================FABOOM文件输入读取结束
    !========================================开始读取网格输入文件geo_information.in
    read(4,*)
    read(4,*)Nkt1
    allocate(Okt1(Nkt1))
    read(4,*)
    read(4,*)Okt1(:)
    
    read(4,*)
    read(4,*)Nkt5
    allocate(Okt5(Nkt5))
    read(4,*)
    read(4,*)Okt5(:)
    
    read(4,*)
    read(4,*)Nkt18
    allocate(Okt18(Nkt18),Ekt18(Nkt18))
    read(4,*)
    read(4,*)Okt18(:),Ekt18(:)
    !write(*,*)Okt18(:),Ekt18(:)
    
    read(4,*)
    read(4,*)Nkt181
    allocate(Okt181(Nkt181),Ekt181(Nkt181))
    read(4,*)
    read(4,*)Okt181(:),Ekt181(:)
    
    read(4,*)
    read(4,*)Nkt20
    allocate(Okt20(Nkt20))
    read(4,*)
    read(4,*)Okt20(:)
    !write(*,*)Okt20(:)
    !write(*,*)Okt1(:)
    
    read(4,*)
    read(4,*)Vtx
    !write(*,*)Vtx
    !读取geo information 完成
    
    
    
    !=================开始读取plot3d中的坐标点并输出
    read(1,*)Nmesh   
    allocate(Nij(Nmesh,2))
    do i=1,Nmesh
        read(1,*)Nij(i,1),Nij(i,2)          !读取网格块数和每块网格的维数
        !write(*,*)Nij(i,1),Nij(i,2)
    end do                                
    !write(6,*)Nmesh
1   format(2(A7))   
2   format(2(F10.5))   
10  format(6(F10.5))  
11  format(8(G10.3))  !2020.8.28修改输出格式
12  format(3(F24.16))    
    
    do i=1,Nmesh
        !=======几何外形dat的文件头========!
        checkface=0
  
        !===============!
        Np=Nij(i,1)*Nij(i,2)
        !write(*,*) Np
        Nrow=int(Np/4)
        Nleft=mod(Np,4)
        !write(*,*)Nrow,Nleft
        allocate(x(Np),y(Np),z(Np))
        x=0
        y=0
        z=0
        if(Nleft==0)then                             !==========分别读xyz，点数是4的倍数
            do j=1,Nrow
                read(1,*)x((j-1)*4+1),x((j-1)*4+2),x((j-1)*4+3),x((j-1)*4+4)
            end do
            do j=1,Nrow
                read(1,*)y((j-1)*4+1),y((j-1)*4+2),y((j-1)*4+3),y((j-1)*4+4)
            end do
            do j=1,Nrow
                read(1,*)z((j-1)*4+1),z((j-1)*4+2),z((j-1)*4+3),z((j-1)*4+4)
            end do                                                        
        else
            do j=1,Nrow
                read(1,*)x((j-1)*4+1),x((j-1)*4+2),x((j-1)*4+3),x((j-1)*4+4)
            end do
            read(1,*)x(Nrow*4+1:Np)
            
            do j=1,Nrow
                read(1,*)y((j-1)*4+1),y((j-1)*4+2),y((j-1)*4+3),y((j-1)*4+4)
            end do
            read(1,*)y(Nrow*4+1:Np)
            do j=1,Nrow
                read(1,*)z((j-1)*4+1),z((j-1)*4+2),z((j-1)*4+3),z((j-1)*4+4)
            end do
            read(1,*)z(Nrow*4+1:Np)
        end if
        
        !write(*,*)"OK"
        !do j=1,Np
        !    write(*,*)z(j)
        !end do
        !write(*,*)x(Np),y(Np),z(Np)
        
        !=================开始判断目前读取的网格属于哪一个边界条件
        !=================开始判断目前读取的网格属于哪一个边界条件
        
        ii=0          !================判断当前网格是不是Kt1边界
20      ii=ii+1
        if(ii<=Nkt1)then
            if(i==Okt1(ii))then
                
                write(2,'(A)')"$points - wing-body  with composite panels","1.","1."
                checkface=1
                goto 30
            else
                goto 20
            end if
        else
            go to 21
        end if
        
21      ii=0        !================判断当前网格是不是Kt5边界
22      ii=ii+1
        if(ii<=Nkt5)then
            if(i==Okt5(ii))then
                
                write(2,'(A)')"$points - bodybase","1.","5."
                checkface=1
                
                goto 30
            else
                goto 22
            endif
        else
            go to 23
        end if

23      ii=0        !================判断当前网格是不是Kt20边界
24      ii=ii+1
        if(ii<=Nkt20)then
            if(i==Okt20(ii))then
                
                write(2,'(A)')"$points - body to wing wakes","1.","20."
                goto 30
            else
                goto 24
            endif
        else
            go to 30
        end if
        
                         
30      continue
        
        !write(*,*)"OK"
        !=====================================判断网格边界条件完成 
        !=====================================判断网格边界条件完成
        write(2,11)Nij(i,1),Nij(i,2)," "," "," "," "," ",i
        Nrow=int(Nij(i,1)/2)
        Nleft=mod(Nij(i,1),2)
        do j=1,Nij(i,2)
            if(Nleft==0)then
                do k=1,Nrow
                    write(2,10)x((j-1)*Nij(i,1)+(k-1)*2+1),y((j-1)*Nij(i,1)+(k-1)*2+1),z((j-1)*Nij(i,1)+(k-1)*2+1),x((j-1)*Nij(i,1)+(k-1)*2+2),y((j-1)*Nij(i,1)+(k-1)*2+2),z((j-1)*Nij(i,1)+(k-1)*2+2)
                   
                    !if(checkface==1)then
                    !    write(6,10)x((j-1)*Nij(i,1)+(k-1)*2+1),y((j-1)*Nij(i,1)+(k-1)*2+1),z((j-1)*Nij(i,1)+(k-1)*2+1)
                    !    write(6,10)x((j-1)*Nij(i,1)+(k-1)*2+2),y((j-1)*Nij(i,1)+(k-1)*2+2),z((j-1)*Nij(i,1)+(k-1)*2+2)
                    !end if
                    
                end do
            else
                do k=1,Nrow
                    write(2,10)x((j-1)*Nij(i,1)+(k-1)*2+1),y((j-1)*Nij(i,1)+(k-1)*2+1),z((j-1)*Nij(i,1)+(k-1)*2+1),x((j-1)*Nij(i,1)+(k-1)*2+2),y((j-1)*Nij(i,1)+(k-1)*2+2),z((j-1)*Nij(i,1)+(k-1)*2+2)
                    
                    !if(checkface==1)then
                    !    write(6,10)x((j-1)*Nij(i,1)+(k-1)*2+1),y((j-1)*Nij(i,1)+(k-1)*2+1),z((j-1)*Nij(i,1)+(k-1)*2+1)
                    !    write(6,10)x((j-1)*Nij(i,1)+(k-1)*2+2),y((j-1)*Nij(i,1)+(k-1)*2+2),z((j-1)*Nij(i,1)+(k-1)*2+2)
                    !end if
                    
                end do
                write(2,10)x((j-1)*Nij(i,1)+Nrow*2+1),y((j-1)*Nij(i,1)+Nrow*2+1),z((j-1)*Nij(i,1)+Nrow*2+1)
                
                !if(checkface==1)then
                !    write(6,10)x((j-1)*Nij(i,1)+Nrow*2+1),y((j-1)*Nij(i,1)+Nrow*2+1),z((j-1)*Nij(i,1)+Nrow*2+1)
                !end if
                
            end if
        enddo
        !=====================================输出几何外形dat（2021.11.25改输出格式精度）
        if(checkface==1)then
            write(6,*)"zone "," i=",Nij(i,1)," j=",Nij(i,2),"k=",1
            do j=1,Nij(i,2)
                if(Nleft==0)then
                    do k=1,Nrow
                        write(6,12)x((j-1)*Nij(i,1)+(k-1)*2+1),y((j-1)*Nij(i,1)+(k-1)*2+1),z((j-1)*Nij(i,1)+(k-1)*2+1)
                        write(6,12)x((j-1)*Nij(i,1)+(k-1)*2+2),y((j-1)*Nij(i,1)+(k-1)*2+2),z((j-1)*Nij(i,1)+(k-1)*2+2)
                    end do
                else
                    do k=1,Nrow
                        write(6,12)x((j-1)*Nij(i,1)+(k-1)*2+1),y((j-1)*Nij(i,1)+(k-1)*2+1),z((j-1)*Nij(i,1)+(k-1)*2+1)
                        write(6,12)x((j-1)*Nij(i,1)+(k-1)*2+2),y((j-1)*Nij(i,1)+(k-1)*2+2),z((j-1)*Nij(i,1)+(k-1)*2+2)
                    end do
                    write(6,12)x((j-1)*Nij(i,1)+Nrow*2+1),y((j-1)*Nij(i,1)+Nrow*2+1),z((j-1)*Nij(i,1)+Nrow*2+1)
                end if
            end do
        end if
        
        !=====================================输出几何外形dat
        deallocate(x,y,z)
    end do                            
    !=========================Plot3d文件读取完，转化坐标点程序完成
    
    !======================================开始写升力面的涡面
    if(Nkt18/=0)then
        write(2,'(A)')"$trailing wakes from wings"!,"1."!2020.8.28修改写入涡面数量
        write(2,"(G10.2)")real(Nkt18)
        write(2,'(A)')"18."
        do i=1,Nkt18
            write(2,11)Okt18(i),Ekt18(i),Vtx,"0."," "," "," ",Nmesh+i
        end do
    end if
    !======================================升力面的涡面信息写完
    
    !======================================开始写base面的涡面信息
    if(Nkt181/=0)then
        write(2,'(A)')"$trailing"!,"1."  !2020.8.28修改写入涡面数量
        write(2,"(G10.2)")real(Nkt181)
        write(2,'(2(G10.2))')"18.","1."
        do i=1,Nkt181
            write(2,11)Okt181(i),Ekt181(i),Vtx,"0."," "," "," ",Nmesh+Nkt18+i
        end do
    end if
    !======================================写base面的涡面信息完成
    
    !======================================开始读写cut信息 2020.9.1添加
    !======================================开始读写cut信息 2020.9.1添加
    !======================================开始读写cut信息 2020.9.1添加
    read(3,*)
    read(3,*)Numcut
    allocate(ycut(Numcut))
    read(3,*)
    read(3,*)ycut(:)
    read(4,*)
    read(4,*)Meshcut
    allocate(ordercut(Meshcut))
    read(4,*)
    read(4,*)ordercut(:)
    read(4,*)
    read(4,*)cutlength,cutratio
    
    if(Numcut/=0)then
    
    write(2,'(A)')"$sectional properties option","1.","*network selection for sectional properties"
    write(2,'(F10.5)')real(meshcut)
    do i=1,Meshcut
        write(2,'(F10.5)')real(ordercut(i))
    end do
    write(2,'(A)')"*cut and reference printout for sectional properties"
    write(2,'(7(F10.1))')1.0,1.0,1.0,1.0,0,1.0,cutlength
    write(2,'(F10.5)')real(Numcut)
    do i=1,Numcut
        write(2,10)0,ycut(i),0,0,1,0
        write(2,'(2(F10.5))')0,cutratio
    end do
    
    end if
    
    
    !======================================开始输入近场提取点
    write(2,'(A)')"$flow","1.        1.","$xyz"
    read(3,*)
    read(3,*)Narea,Nnear
    !write(temp1,*)Nnear

    x0=0
    x1=lm
    write(2,'(F10.5)')real(Nnear)
    allocate(near(Nnear,3))
    xs=x0+(x1-x0)*R*(Mach**2-1)**0.5-L
    xe=x1+(x1-x0)*R*(Mach**2-1)**0.5+3*L!这里延长信号长度避免A502直接输出的错误
    do i=1,Nnear
        near(i,1)=xs+(xe-xs)/(Nnear-1)*(i-1) 
        near(i,2)=lm*R*sin(phi/180*pai)
        near(i,3)=-lm*R*cos(phi/180*pai)
        !write(3,10)near(i,1),near(i,2),near(i,3)
    end do
    do i=1,Nnear/2
        write(2,10)near(2*i-1,1),near(2*i-1,2),near(2*i-1,3),near(2*i,1),near(2*i,2),near(2*i,3)
    end do
    !======================================完成输入近场提取点
    
    !======================================输入A502输入文件的结束标记
    write(2,'(A)')"$end of a502 inputs"
    close(unit=1)
    close(unit=2)
    close(unit=3)
    close(unit=4)
    !close(unit=5)
    close(unit=6)
    end subroutine A502in
    
    

    subroutine A502in_2  
    implicit none
    
    character(len=20)::filename,str
    character string
    integer i,j,k,ii,jj,kk
    integer Nmesh,Np,Nrow,Nleft,Numcut,meshcut,Ni,Nj
    integer Nkt1,Nkt5,Nkt18,Nkt181,Nkt20
    integer Narea,Nnear
    integer checkkt20,checkfw
    integer,allocatable::Okt1(:),Okt5(:),Okt18(:),Okt181(:),Ekt18(:),Ekt181(:),Ordercut(:)
    integer,allocatable::OKt20A(:),OKt20B(:),OKt20C(:),OKt20D(:)
    real(kind=8),allocatable::XKt20A(:,:),XKt20B(:,:),XKt20c(:,:),XKt20D(:,:)
    real(kind=8),allocatable::YKt20A(:,:),YKt20B(:,:),YKt20c(:,:),YKt20D(:,:)
    real(kind=8),allocatable::ZKt20A(:,:),ZKt20B(:,:),ZKt20c(:,:),ZKt20D(:,:)
    real(kind=8),allocatable::Xlastpoint(:),ylastpoint(:),zlastpoint(:)
    integer,allocatable::kt20dm(:)
    real(kind=8) temp,mach,AOA,H,rou,p,R,PHI,checkface
    real (kind=8) L,Lm,S
    real(kind=8) Vtx
    real(kind=8) cutlength,cutratio
    real(kind=8) x0,x1,xs,xe
    real::pai=3.141592658
    integer,allocatable::Nij(:,:)
    real(kind=8),allocatable::x(:),y(:),z(:),ycut(:)
    real(kind=8),allocatable::near(:,:)
    !write(*,*)'输入文件名'
    !read(*,*)filename
    !write(*,*)filename
    open(unit=1,file="indata/NEW-SHAPE.dat",action='read',status='old')            !!!!2020.6.19修改输入文件至indata
    open(unit=2,file="A502/A502.in",action="write",status="replace")
    open(unit=3,file="indata/FABoom.in",action='read',status='old')
    open(unit=4,file="indata/geo_information2.in",action='read',status='old')
    !open(unit=5,file="indata/atmospheric_profile.in",action='read',status='old')
    open(unit=6,file="tempdata/geometry.dat",action='write',status='replace')
   
    !================================!!子程序A502in变量说明!!!==================================！
    !Nmesh,Np,Nrow,Nleft               网格块数，当前网格的网格点数，数据块的行数，数据最后一行的数据个数
    !Nkt1,Nkt5,Nkt18,Nkt181,Nkt20      kt1边界网格数，kt5边界网格数，kt18边界网格数，kt18.1边界网格数，kt20边界网格数
    !Narea,Nnear                       面积分布中所用的点数，近场信号提取的点数
    !Okt1,Okt5,Okt18,Okt181,Okt20      kt1网格编号，kt5网格编号，kt18网格编号，kt18.1网格编号，kt20网格编号
    !Ekt18,Ekt181                      kt18网格搭接尾迹的edge编号，kt18.1网格搭接尾迹的edge编号     
    !temp,mach,AOA,H,rou,p,R,PHI       活动变量，马赫数，迎角，飞行高度，密度，无穷来流静压，提取位置的体长倍数，周向角
    !L,Lm,S                            飞机长度，计算模型长度，半模参考面积
    !Vtx                               尾迹延伸末尾的x坐标
    !x0,x1,xs,xe                       飞机头部x坐标，飞机尾部x坐标，近场提取点开始坐标，近场提取点结束坐标
    !Nij,x,y,z,near                    网格维数数组，x坐标，y坐标，z坐标，近场提取点坐标数组
    !checkface                         检查是物面网格的标记
    !================================!!子程序A502in变量说明!!!==================================！
    
    
    
    !========================================开始读取FABOOM，写A502in文件开头
    
    read(3,*)str,str,str
    write(2,1)"$TITLE ",str
    read(3,*)str
    write(2,'(A)')str,"Created by PANIN","$DATACHECK","0.","$symmetry - xz plane"
    write(2,'(2(A10))')"1.","0."
    write(2,'(A)')"$mach number"
    read(3,*)
    read(3,*)mach,AOA,H,rou,p,R,PHI
    write(2,'(F10.3)')mach
   
    !write(2,'(A)')"$cases - no. of solutions","1.","$angles-of-attack","0."
    write(2,'(A)')"$cases - no. of solutions","1.","$angles-of-attack"
    write(2,'(F10.5)')AOA
    write(2,'(F10.5)')AOA
    write(2,'(A)')"$printout options","0.        0.        0.        1.        0.        0.",".0        0.        0.        0.        0."
    write(2,'(A)')"$references for accumulated forces and moments","0.        0.        0."
    
    read(3,*)
    read(3,*)L,Lm,temp,S
   
    !write(str,*)S
    
    write(2,'(F10.6,3(A10))')S,"        1.","        1.","        0."    
         
    !========================================FABOOM文件输入读取结束
    !========================================开始读取网格输入文件geo_information.in
    read(4,*)
    read(4,*)Nkt1,Nmesh
    allocate(Okt1(Nkt1))
    read(4,*)
    read(4,*)Okt1(:)
    
    read(4,*)
    read(4,*)Nkt5
    allocate(Okt5(Nkt5))
    read(4,*)
    read(4,*)Okt5(:)
    
    read(4,*)
    read(4,*)Nkt18
    allocate(Okt18(Nkt18),Ekt18(Nkt18))
    read(4,*)
    read(4,*)Okt18(:),Ekt18(:)
    !write(*,*)Okt18(:),Ekt18(:)
    
    read(4,*)
    read(4,*)Nkt181
    allocate(Okt181(Nkt181),Ekt181(Nkt181))
    read(4,*)
    read(4,*)Okt181(:),Ekt181(:)
    
    read(4,*)
    read(4,*)Nkt20
    allocate(xlastpoint(Nkt20),ylastpoint(Nkt20),zlastpoint(Nkt20),kt20dm(Nkt20))
    allocate(OKt20A(2),OKt20B(2),OKt20C(2),OKt20D(2))
    read(4,*)
    if(Nkt20/=0)then
    do i=1,Nkt20
        if(i==1)then
            read(4,*)Okt20A(:)
        end if
        if(i==2)then
            read(4,*)Okt20B(:)
        end if
        if(i==3)then
            read(4,*)Okt20C(:)
        end if
        if(i==4)then
            read(4,*)Okt20D(:)
        end if
    end do
    else
        read(4,*)
    end if
    
    
    !write(*,*)Okt20(:)
    !write(*,*)Okt1(:)
    
    read(4,*)
    read(4,*)Vtx
    !write(*,*)Vtx
    !读取geo information 完成
   ! write(*,*)"ok"
    
    
    !=================开始读取plot3d中的坐标点并输出
    !read(1,*)Nmesh   
    
    !do i=1,Nmesh
    !    read(1,*)Nij(i,1),Nij(i,2)          !读取网格块数和每块网格的维数
    !    !write(*,*)Nij(i,1),Nij(i,2)
    !end do                                
    !write(6,*)Nmesh
1   format(2(A7))   
2   format(2(F10.5))   
10  format(6(F10.5))  
11  format(8(G10.3))  !2020.8.28修改输出格式
    
    do i=1,Nmesh
        !=======几何外形dat的文件头========!
        checkface=0
        read(1,*)string,string,Ni,string,Nj
        Np=Ni*Nj
        allocate(x(Np),y(Np),z(Np))
        do j=1,Np
            read(1,*)x(j),y(j),z(j)
        end do

        
        !=================开始判断目前读取的网格属于哪一个边界条件
        !=================开始判断目前读取的网格属于哪一个边界条件
        
        ii=0          !================判断当前网格是不是Kt1边界
20      ii=ii+1
        if(ii<=Nkt1)then
            if(i==Okt1(ii))then
                
                write(2,'(A)')"$points - wing-body  with composite panels","1.","1."
                checkface=1
                goto 30
            else
                goto 20
            end if
        else
            go to 21
        end if
        
21      ii=0        !================判断当前网格是不是Kt5边界
22      ii=ii+1
        if(ii<=Nkt5)then
            if(i==Okt5(ii))then
                
                write(2,'(A)')"$points - bodybase","1.","5."
                checkface=1
                
                goto 31
            else
                goto 22
            endif
        else
            go to 31
        end if
30     continue 
23      ii=0        !================判断当前网格是不是Kt20边界
24      ii=ii+1
        if(ii<=Nkt20)then
            if(ii==1)then
                if(i==Okt20A(1))then
                    allocate(XKt20A(2,Nj+1),YKt20A(2,Nj+1),ZKt20A(2,Nj+1))
                    kt20dm(ii)=Nj+1
                    do jj=1,Nj
                       XKt20A(1,jj)= x(jj*Ni)
                       YKt20A(1,jj)= y(jj*Ni)
                       ZKt20A(1,jj)= z(jj*Ni)
                    end do
                    XKt20A(1,Nj+1)=vtx
                    YKt20A(1,Nj+1)=YKt20A(1,Nj)
                    ZKt20A(1,Nj+1)=ZKt20A(1,Nj)
                    goto 31
                end if
                if(i==Okt20A(2))then
                    xlastpoint(ii)=x(Np)
                    ylastpoint(ii)=y(Np)
                    zlastpoint(ii)=z(Np)
                    goto 31
                end if
                goto 24
            end if
            
            if(ii==2)then
                if(i==Okt20B(1))then
                    allocate(XKt20B(2,Nj+1),YKt20B(2,Nj+1),ZKt20B(2,Nj+1))
                    kt20dm(ii)=Nj+1
                    do jj=1,Nj
                       XKt20B(1,jj)= x(jj*Ni)
                       YKt20B(1,jj)= y(jj*Ni)
                       ZKt20B(1,jj)= z(jj*Ni)
                    end do
                    XKt20B(1,Nj+1)=vtx
                    YKt20B(1,Nj+1)=YKt20B(1,Nj)
                    ZKt20B(1,Nj+1)=ZKt20B(1,Nj)
                    goto 31
                end if
                if(i==Okt20B(2))then
                    xlastpoint(ii)=x(Np)
                    ylastpoint(ii)=y(Np)
                    zlastpoint(ii)=z(Np)
                    goto 31
                end if
                goto 24
            end if
            
            if(ii==3)then
                if(i==Okt20C(1))then
                    allocate(XKt20C(2,Nj+1),YKt20C(2,Nj+1),ZKt20C(2,Nj+1))
                    kt20dm(ii)=Nj+1
                    do jj=1,Nj
                       XKt20C(1,jj)= x(jj*Ni)
                       YKt20C(1,jj)= y(jj*Ni)
                       ZKt20C(1,jj)= z(jj*Ni)
                    end do
                    XKt20C(1,Nj+1)=vtx
                    YKt20C(1,Nj+1)=YKt20C(1,Nj)
                    ZKt20C(1,Nj+1)=ZKt20C(1,Nj)
                    goto 31
                end if
                if(i==Okt20C(2))then
                    xlastpoint(ii)=x(Np)
                    ylastpoint(ii)=y(Np)
                    zlastpoint(ii)=z(Np)
                    goto 31
                end if
                goto 24
            end if
            
            if(ii==4)then
                if(i==Okt20D(1))then
                    allocate(XKt20D(2,Nj+1),YKt20D(2,Nj+1),ZKt20D(2,Nj+1))
                    kt20dm(ii)=Nj+1
                    do jj=1,Nj
                       XKt20D(1,jj)= x(jj*Ni)
                       YKt20D(1,jj)= y(jj*Ni)
                       ZKt20D(1,jj)= z(jj*Ni)
                    end do
                    XKt20D(1,Nj+1)=vtx
                    YKt20D(1,Nj+1)=YKt20D(1,Nj)
                    ZKt20D(1,Nj+1)=ZKt20D(1,Nj)
                    goto 31
                end if
                if(i==Okt20D(2))then
                    xlastpoint(ii)=x(Np)
                    ylastpoint(ii)=y(Np)
                    zlastpoint(ii)=z(Np)
                    goto 31
                end if
                goto 24
            end if
        else
            go to 31
        end if
31      continue            
            
            !
            !if(i==Okt20(ii))then
            !    
            !    write(2,'(A)')"$points - body to wing wakes","1.","20."
            !    goto 30
            !else
            !    goto 24
            !endif
       
        
                         

        
        !write(*,*)"OK"
        !=====================================判断网格边界条件完成 
        !=====================================判断网格边界条件完成
        write(2,11)Ni,Nj," "," "," "," "," ",i
        Nrow=int(Ni/2)
        Nleft=mod(Ni,2)
        do j=1,Nj
            if(Nleft==0)then
                do k=1,Nrow
                    write(2,10)x((j-1)*Ni+(k-1)*2+1),y((j-1)*Ni+(k-1)*2+1),z((j-1)*Ni+(k-1)*2+1), &
                    x((j-1)*Ni+(k-1)*2+2),y((j-1)*Ni+(k-1)*2+2),z((j-1)*Ni+(k-1)*2+2)
                end do
            else
                do k=1,Nrow
                    write(2,10)x((j-1)*Ni+(k-1)*2+1),y((j-1)*Ni+(k-1)*2+1),z((j-1)*Ni+(k-1)*2+1), &
                    x((j-1)*Ni+(k-1)*2+2),y((j-1)*Ni+(k-1)*2+2),z((j-1)*Ni+(k-1)*2+2)
                end do
                write(2,10)x((j-1)*Ni+Nrow*2+1),y((j-1)*Ni+Nrow*2+1),z((j-1)*Ni+Nrow*2+1)                
            end if
        enddo
        
        
        
        !=====================================输出几何外形dat
        if(checkface==1)then
            write(6,*)"zone "," i=",Ni," j=",Nj,"k=1"
            do j=1,Nj
                if(Nleft==0)then
                    do k=1,Nrow
                        write(6,10)x((j-1)*Ni+(k-1)*2+1),y((j-1)*Ni+(k-1)*2+1),z((j-1)*Ni+(k-1)*2+1)
                        write(6,10)x((j-1)*Ni+(k-1)*2+2),y((j-1)*Ni+(k-1)*2+2),z((j-1)*Ni+(k-1)*2+2)
                    end do
                else
                    do k=1,Nrow
                        write(6,10)x((j-1)*Ni+(k-1)*2+1),y((j-1)*Ni+(k-1)*2+1),z((j-1)*Ni+(k-1)*2+1)
                        write(6,10)x((j-1)*Ni+(k-1)*2+2),y((j-1)*Ni+(k-1)*2+2),z((j-1)*Ni+(k-1)*2+2)
                    end do
                    write(6,10)x((j-1)*Ni+Nrow*2+1),y((j-1)*Ni+Nrow*2+1),z((j-1)*Ni+Nrow*2+1)
                end if
            end do
        end if
        
        !=====================================输出几何外形dat
        deallocate(x,y,z)
    end do                            
    !=========================Plot3d文件读取完，转化坐标点程序完成
    !=========================Plot3d文件读取完，转化坐标点程序完成
    !=========================Plot3d文件读取完，转化坐标点程序完成
    
    !=====================================写Kt20
    !write(2,'(A)')"$points - body to wing wakes","1.","20."
    do i=1,NKt20
        if(i==1)then
            do j=2,kt20dm(i)
                XKt20A(2,j)=XKt20A(1,j)
                YKt20A(2,j)=ylastpoint(i)
                ZKt20A(2,j)=zlastpoint(i)
            end do
            XKt20A(2,1)=xlastpoint(i)
            YKt20A(2,1)=ylastpoint(i)
            ZKt20A(2,1)=zlastpoint(i)
            write(2,'(A)')"$points - body to wing wakes","1.","20."
            write(2,11)2,kt20dm(i)," "," "," "," "," ",Nmesh+i
            do j=1,kt20dm(i)
                write(2,10)XKt20A(1,j),YKt20A(1,j),ZKt20A(1,j),XKt20A(2,j),YKt20A(2,j),ZKt20A(2,j)
            end do
        end if
        
        if(i==2)then
            do j=2,kt20dm(i)
                XKt20B(2,j)=XKt20B(1,j)
                YKt20B(2,j)=ylastpoint(i)
                ZKt20B(2,j)=zlastpoint(i)
            end do
            XKt20B(2,1)=xlastpoint(i)
            YKt20B(2,1)=ylastpoint(i)
            ZKt20B(2,1)=zlastpoint(i)
            write(2,'(A)')"$points - body to wing wakes","1.","20."
            write(2,11)2,kt20dm(i)," "," "," "," "," ",Nmesh+i
            do j=1,kt20dm(i)
                write(2,10)XKt20B(1,j),YKt20B(1,j),ZKt20B(1,j),XKt20B(2,j),YKt20B(2,j),ZKt20B(2,j)
            end do
        end if
            
        if(i==3)then
            do j=2,kt20dm(i)
                XKt20C(2,j)=XKt20C(1,j)
                YKt20C(2,j)=ylastpoint(i)
                ZKt20C(2,j)=zlastpoint(i)
            end do
            XKt20C(2,1)=xlastpoint(i)
            YKt20C(2,1)=ylastpoint(i)
            ZKt20C(2,1)=zlastpoint(i)
            write(2,'(A)')"$points - body to wing wakes","1.","20."
            write(2,11)2,kt20dm(i)," "," "," "," "," ",Nmesh+i
            do j=1,kt20dm(i)
                write(2,10)XKt20C(1,j),YKt20C(1,j),ZKt20C(1,j),XKt20C(2,j),YKt20C(2,j),ZKt20C(2,j)
            end do
        end if
        
        if(i==4)then
            do j=2,kt20dm(i)
                XKt20D(2,j)=XKt20D(1,j)
                YKt20D(2,j)=ylastpoint(i)
                ZKt20D(2,j)=zlastpoint(i)
            end do
            XKt20D(2,1)=xlastpoint(i)
            YKt20D(2,1)=ylastpoint(i)
            ZKt20D(2,1)=zlastpoint(i)
            write(2,'(A)')"$points - body to wing wakes","1.","20."
            write(2,11)2,kt20dm(i)," "," "," "," "," ",Nmesh+i
            do j=1,kt20dm(i)
                write(2,10)XKt20D(1,j),YKt20D(1,j),ZKt20D(1,j),XKt20D(2,j),YKt20D(2,j),ZKt20D(2,j)
            end do
        end if
    end do
    
    
    
    !======================================开始写升力面的涡面
    if(Nkt18/=0)then
        write(2,'(A)')"$trailing wakes from wings"!,"1."!2020.8.28修改写入涡面数量
        write(2,"(G10.2)")real(Nkt18)
        write(2,'(A)')"18."
        do i=1,Nkt18
            write(2,11)Okt18(i),Ekt18(i),Vtx,"0."," "," "," ",Nmesh+Nkt20+i
        end do
    end if
    !======================================升力面的涡面信息写完
    
    !======================================开始写base面的涡面信息
    if(Nkt181/=0)then
        write(2,'(A)')"$trailing"!,"1."  !2020.8.28修改写入涡面数量
        write(2,"(G10.2)")real(Nkt181)
        write(2,'(2(G10.2))')"18.","1."
        do i=1,Nkt181
            write(2,11)Okt181(i),Ekt181(i),Vtx,"0."," "," "," ",Nmesh+Nkt20+Nkt18+i
        end do
    end if
    !======================================写base面的涡面信息完成
    
    !======================================开始读写cut信息 2020.9.1添加
    !======================================开始读写cut信息 2020.9.1添加
    !======================================开始读写cut信息 2020.9.1添加
    read(3,*)
    read(3,*)Numcut
    allocate(ycut(Numcut))
    read(3,*)
    read(3,*)ycut(:)
    read(4,*)
    read(4,*)Meshcut
    allocate(ordercut(Meshcut))
    read(4,*)
    read(4,*)ordercut(:)
    read(4,*)
    read(4,*)cutlength,cutratio
    
    if(Numcut/=0)then
    
    write(2,'(A)')"$sectional properties option","1.","*network selection for sectional properties"
    write(2,'(F10.5)')real(meshcut)
    do i=1,Meshcut
        write(2,'(F10.5)')real(ordercut(i))
    end do
    write(2,'(A)')"*cut and reference printout for sectional properties"
    write(2,'(7(F10.1))')1.0,1.0,1.0,1.0,0,1.0,cutlength
    write(2,'(F10.5)')real(Numcut)
    do i=1,Numcut
        write(2,10)0,ycut(i),0,0,1,0
        write(2,'(2(F10.5))')0,cutratio
    end do
    
    end if
    
    
    !======================================开始输入近场提取点
    write(2,'(A)')"$flow","1.        1.","$xyz"
    read(3,*)
    read(3,*)Narea,Nnear
    !write(temp1,*)Nnear

    x0=0
    x1=lm
    write(2,'(F10.5)')real(Nnear)
    allocate(near(Nnear,3))
    xs=x0+(x1-x0)*R*(Mach**2-1)**0.5-L
    xe=x1+(x1-x0)*R*(Mach**2-1)**0.5+3*L!这里延长信号长度避免A502直接输出的错误
    do i=1,Nnear
        near(i,1)=xs+(xe-xs)/(Nnear-1)*(i-1) 
        near(i,2)=lm*R*sin(phi/180*pai)
        near(i,3)=-lm*R*cos(phi/180*pai)
        !write(3,10)near(i,1),near(i,2),near(i,3)
    end do
    do i=1,Nnear/2
        write(2,10)near(2*i-1,1),near(2*i-1,2),near(2*i-1,3),near(2*i,1),near(2*i,2),near(2*i,3)
    end do
    !======================================完成输入近场提取点
    
    !======================================输入A502输入文件的结束标记
    write(2,'(A)')"$end of a502 inputs"
    close(unit=1)
    close(unit=2)
    close(unit=3)
    close(unit=4)
    !close(unit=5)
    close(unit=6)
    
    
    end subroutine A502in_2
    
    !===================================================================================================！
    !                                                                                                   ！
    !                读取panair.out文件生成升力等效截面积和直接近场信号的子程序A502out                  ！
    !                读取panair.out文件生成升力等效截面积和直接近场信号的子程序A502out                  ！
    !                读取panair.out文件生成升力等效截面积和直接近场信号的子程序A502out                  ！
    !                                 2020.9.1添加机翼剖面信息的输出                                    ！    
    !===================================================================================================！    
    
    subroutine A502out1
    implicit none
    integer i,j,k,ii,jj,kk,Nmesh,Nfile,Nele,Nelei,Nlast,NN,Nnear,Numcut,Sec_P,Nummesh   !Nmesh网格块数，Nfile流场数据读取的段数，Nele总面元数，Nelei单块网格面元数,Nlast最后一段数据的面元数,NN 面元循环累计计数器,Nnear近场点数
    real mach,Sref,AOA,X0,xnear,dpp,Total                         !X0用于坐标转换,dpp是dp/p
    real miu,chord,Reno,fm,cfi,cdf,cdp,cl,ld,Swet,cd,cm,xref
    integer,allocatable::Nij(:,:)              !Nij各块网格的维度数，该数组为Nmesh*2维
    integer,allocatable::Ordermesh(:)  !做剖面信息的网格编号
    real,allocatable:: x(:),y(:),z(:),nx(:),ny(:),nz(:),Cp(:),secx(:),secy(:),secz(:),secx1(:),seccp(:)
    real,allocatable::Lift(:,:),Larea(:,:)!,ox(:,:),oy(:,:),oz(:,:),ocp(:,:)      !lift升力分布，Larea升力面积分布
    real,allocatable::Fz(:,:),Ftran(:,:)        !Fz每个面元的升力,Ftran转换X坐标的升力数组
    real,allocatable::Ylift(:,:),spanlift(:,:) 
    real::H=-30.0                               !用于进行轴向坐标转换的位置
    real rou,p0,B
    real ox,oy,oz,ocp
    real::c=295.07
    real::pi=3.141592658
    real tem                                    !临时数据，用于赋值不需要的数据
!    real int                                    !临时整形数据，用于赋值不需要的数据
    real::gamma=1.4
    integer chazhi
    character(len=256)::string1                
    character(len=10)::string2
    
    !======================================!!子程序A502out变量说明!!!=========================================！    
    !Nmesh,Nfile，Nele                        网格块数，数据的段数，总面元数！
    !Nelei，Nlast,NN,Nnear                    单块网格面元数,最后一段数据的面元数，面元循环累计计数器,近场点数！  
    !mach,Sref,AOA,X0,xnear,dpp,Total         马赫数，半模参考面积，迎角，坐标转换x，近场点数，dp/p，总升力   ！
    !Nij(:,:)                                 Nij各块网格的维度数，该数组为Nmesh*2维                          ！
    !x(:),y(:),z(:),nx(:),ny(:),nz(:),Cp(:)   面元中心点XYZ坐标及其法向量，压力系数                           ！
    !Lift(:,:),Larea(:,:)                     升力分布，升力面积分布                                          !
    !Fz(:,:),Ftran(:,:)                       每个面元的升力,转换X坐标的升力数组                              !
    !chazhi                                   等效截面积插值点数                                              ！
    !======================================!!子程序A502out变量说明!!!=========================================！
    
    open(unit=1,FILE="A502/panair.out",action="read",status="old")
    open(unit=2,FILE="A502/Lift distribution.dat",action="write",status="replace")
    open(unit=3,FILE="A502/Lift area.dat",action="write",status="replace")
    open(unit=4,FILE="indata/FABoom.in",action="read",status="old")
    open(unit=5,FILE="A502/A502 nearfield signal.dat",action="write",status="replace")
    open(unit=6,FILE="A502/agps",action="read",status="old")
    open(unit=7,FILE="indata/geo_information1.in",action="read",status="old")
    !
    open(unit=9,FILE="A502/cut.dat",action="write",status="replace")
    open(unit=10,FILE="A502/surface_Cp.dat",action="write",status="replace")
    !write(*,*)"          000000     0           00000   0000   0000  0     0"
    !write(*,*)"          0         0 0          0    0 0    0 0    0 00   00"
    !Write(*,*)"          000000   0   0  00000  00000  0    0 0    0 0 0 0 0"
    !Write(*,*)"          0       0000000        0    0 0    0 0    0 0  0  0"
    !Write(*,*)"          0      0       0       00000   0000   0000  0     0"

    !write(*,*)"输入密度, 压强,插值点数"
    !read(*,*)rou
    !read(*,*)p0
    !read(*,*)chazhi
!=======================================================================读取inp文件    
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*) 
      
    read(4,*)tem,AOA,tem,rou,p0
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*)Numcut
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*)chazhi
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*)miu,chord,xref
    
   
    
!=======================================================================读取马赫数    
1   read(1,'(A)')string1
    if (index(string1,"$mach number")/= 0)then
        go to 2
    else
        go to 1
    end if
2   continue    
    read(1,*)tem,Mach     
    
!========================================================================读取参考面积    
3   read(1,'(A)')string1
    if (index(string1,"$references for accumulated forces and moments")/= 0)then
        go to 4
    else
        go to 3
    end if
4   continue    
    read(1,*)
    read(1,*)tem,Sref  
    
!=========================================================================读取网格块数   
!5   read(1,'(A)')string1
!    if (index(string1,"$points - wing-body  with composite panels")/= 0)then
!        go to 6
!    else
!        go to 5
!    end if
!6   continue    
    read(7,*)
    read(7,*)Nmesh,Nmesh                                 
    !write(*,*)Mach,Sref,Nmesh
  
!=========================================================================读取近场点数
13  read(1,'(A)')string1
    if (index(string1,"$xyz")/= 0)then
        go to 14
    else
        go to 13
    end if
14  continue
    read(1,*)tem,Nnear
    
!=========================================================================读取每块网格面元维数(不是网格维数）   
7   read(1,'(A)')string1
    if (index(string1,"network id&index   #rows   #cols")/= 0)then
        go to 8
    else
        go to 7
    end if
8   continue   
    read(1,*)
    allocate(Nij(Nmesh,2))
    do i=1,Nmesh
        read(1,*)string2,tem,Nij(i,1),Nij(i,2)
        Nij(i,1)=Nij(i,1)-1
        Nij(i,2)=Nij(i,2)-1
        !write(*,*)Nij(i,:)
    end do
   
 !=========================================================================计算总面元数
    Nele=Nij(1,1)*Nij(1,2)
    do i=2,Nmesh
        Nele=Nele+Nij(i,1)*Nij(i,2)
        !write(*,*)Nele
    end do
    !write(*,*)Nele
    
    
!=========================================================================建立读取面元流场信息的数组x,y,z,nx,ny,nz,cp
    allocate(x(Nele),y(Nele),z(Nele),nx(Nele),ny(Nele),nz(Nele),Cp(Nele))
9   read(1,'(A)')string1
    if (index(string1,"simultaneous solution number")/= 0)then
        go to 10
    else
        go to 9
    end if
10  continue 
    !write(*,*)string1
!=========================================================================开始读取流场信息
!=========================================================================开始读取流场信息   
!=========================================================================开始读取流场信息    
    NN=1
     
    do i=1,Nmesh                                   !大循环是网格块数，读取块数次网格
        Nelei=Nij(i,1)*Nij(i,2)                    !求本块网格面元数
        Nfile=int(Nelei/10)
        Nlast=mod(Nelei,10)
       ! write(*,*)Nfile,Nlast
11       read(1,'(A)')string1
        if (index(string1,"0*e*for-mom")/= 0.OR.index(string1,"freestream velocity")/= 0)then  !寻找数据开始特征字段
            go to 12
        else
            go to 11
        end if
12      continue 
        !write(*,*)"牛批"

        do j=1,Nfile                               !循环数据段数
            !write(*,*)"牛批",j
            do ii=1,7
                read(1,*)! string1
            end do
            !write(*,*)string1
            do jj=1,10                        !循环每段下面10段数据
                read(1,*)tem,tem,x(NN),y(NN),z(NN),tem,tem,tem,tem,tem,nx(NN),ny(NN),nz(NN)
                read(1,'(11E11.4)')tem,tem,tem,tem,tem,tem,tem,tem,tem,tem,Cp(NN)
                read(1,*)
                read(1,*)
                !write(*,*)NN,Cp(NN)!NN,x(NN),y(NN),z(NN),nx(NN),ny(NN),nz(NN),
                NN=NN+1
                !write(*,*)i,j,jj
            end do
            !write(*,*)"牛批",j
        end do
        
        if(Nlast>0)then      !对非10倍数的Nelei读取剩下的数据
            do ii=1,7
                read(1,*)
            end do
            do jj=1,Nlast                        !循环余下的Nlast段数据
                read(1,*)tem,tem,x(NN),y(NN),z(NN),tem,tem,tem,tem,tem,nx(NN),ny(NN),nz(NN)
                read(1,'(11E11.4)')tem,tem,tem,tem,tem,tem,tem,tem,tem,tem,Cp(NN)    
                read(1,*)
                read(1,*)
                !write(*,*)NN,x(NN)!,y(NN),z(NN),nx(NN),ny(NN),nz(NN),
                NN=NN+1
            end do
        end if
         !write(*,*)"牛批"
        !write(*,*)NN,"mesh",i,"done"
    end do
    !write(*,*)Cp(Nele)
    !write(*,*)"牛批"
    !write(*,*)"OK"
!=========================================================================流场信息读取完成
!=========================================================================流程信息读取完成   
!=========================================================================流程信息读取完成
        
!=========================================================================计算并赋值面元升力数组
    allocate(Fz(Nele,3))                              !Nele列，3维，第一列x，第二列z，第三列Fz
    !do i=1,Nele
    !    Fz(i,1)=x(i)
    !    Fz(i,2)=z(i)
    !    Fz(i,3)=-Cp(i)*0.5*rou*(Mach*c)**2*nz(i)*Sref
    !end do
    !!write(*,*)Fz(Nele,:)
    !Total=0
    !do i=1,Nele
    !    Total=Fz(i,3)+Total
    !end do
  
!    
!!==========================================================================不坐标转换
!    allocate(Ftran(Nele,2))
!
!    do i=1,Nele
!        Ftran(i,1)=x(i)  
!        Ftran(i,2)=-Cp(i)*0.5*rou*(Mach*c)**2*nz(i)*Sref
!        !write(*,*)Ftran(i,2)
!    end do

!=========================================================================坐标转换  
!allocate(Fz(Nele,3))                              !Nele列，3维，第一列x，第二列z，第三列Fz
    do i=1,Nele
        Fz(i,1)=cos(AOA*pi/180)*x(i)+sin((AOA*pi/180))*z(i)          !旋转网格
        Fz(i,2)=-sin(AOA*pi/180)*x(i)+cos((AOA*pi/180))*z(i)
        Fz(i,3)=-Cp(i)*0.5*rou*(Mach*c)**2*nz(i)*Sref
    end do
    !write(*,*)Fz(Nele,:)
    Total=0
    do i=1,Nele
        Total=Fz(i,3)+Total
    end do
    write(*,*)"重量为",total*0.2,"kg" 
    
    allocate(Ftran(Nele,2))
    do i=1,Nele
        
        Ftran(i,1)=Fz(i,1)+(Fz(i,2)-H)*(Mach**2-1)**0.5    
        Ftran(i,2)=Fz(i,3)
        
    end do
!=========================================================================对X排序升力数组  
    do i=1,Nele-1
        do j=i+1,Nele
            if(Ftran(j,1)<Ftran(i,1))then
                tem=Ftran(i,1)
                Ftran(i,1)=Ftran(j,1)
                Ftran(j,1)=tem
                tem=Ftran(i,2)
                Ftran(i,2)=Ftran(j,2)
                Ftran(j,2)=tem
            end if
        end do
    end do
    !do i=1,Nele
        !write(*,*)Fz(1,1)! Ftran(1,1)!,Ftran(i,1)
    !end do
!=========================================================================升力分布求和
    allocate(Lift(Nele,2))
    tem=Ftran(1,1)
    Lift(1,1)=0!Ftran(1,1)! 2020 9.7  Fz(1,1)
   ! Lift(1,2)=Ftran(1,2)
    Lift(1,2)=0 !赋值起点为零
    !write(*,*)lift(1,1),lift(1,2)
    do i=2,Nele
        Lift(i,1)=Ftran(i,1)-tem!+Fz(1,1)!(Fz(1,2)-H)*(Mach**2-1)**0.5!
        Lift(i,2)=Lift(i-1,2)+Ftran(i,2)
        !write(2,*)Lift(i,1),Lift(i,2)
    end do
    !write(*,*)Lift(Nele,2)*2
!=========================================================================升力分布插值    
    allocate(Larea(chazhi,2))
    do i=1,chazhi
        Larea(i,1)=(Lift(Nele,1)-Lift(1,1))/(chazhi-1)*(i-1)+Lift(1,1)
    end do
    Larea(1,2)=Lift(1,2)*2
    Larea(chazhi,2)=Lift(Nele,2)*2
    write(2,*)Larea(1,1),Larea(1,2)
    do i=2,chazhi-1
        do j=1,Nele-1
            if(Larea(i,1)>Lift(j,1).AND.Larea(i,1)<=Lift(j+1,1))then
                Larea(i,2)=Lift(j,2)+(Lift(j+1,2)-Lift(j,2))/(Lift(j+1,1)-Lift(j,1))*(Larea(i,1)-Lift(j,1))
            end if
        end do
        Larea(i,2)=Larea(i,2)*2
        write(2,*)Larea(i,1),Larea(i,2)
    end do
    write(2,*)Larea(chazhi,1),Larea(chazhi,2)
    !write(*,*)Larea(i,1),Larea(chazhi,2)
!=========================================================================计算升力等效截面积分布    
    B=(Mach**2-1)**0.5
    write(3,*)chazhi+2!"VARIABLES=X,S<sub>L</sub>(x)"
    Write(3,*)0,0
    do i=1,chazhi
        Larea(i,2)=Larea(i,2)*B/(rou*(Mach*c)**2)!(gamma*p0*Mach**2)
        write(3,*)Larea(i,:)
    end do
    write(3,*)Larea(chazhi,1)*1.2,Larea(chazhi,2)    !!!2020.11.4改长度
    
    
!============================================================================================读取输出dp/p    
15   read(1,'(A)')string1
    if (index(string1,"off body flow characteristics")/= 0)then
        go to 16
    else
        go to 15
    end if
16  continue
    !write(*,*)"niupi"
    read(1,*)
    read(1,*)
    read(1,*)
    write(5,*)"VARIABLES=X,dp/p"
    do i=1,Nnear
        read(1,*)tem,tem,xnear,tem,tem,tem,tem,tem,tem,dpp
        dpp=dpp*0.5*rou*(Mach*c)**2/p0
        write(5,*)xnear,dpp
    end do
    
    
    



!============================================================================================输出机翼剖面信息 2020.9.2修改了输出剖面的bug
17   read(7,'(A)')string1
    if (index(string1,"mesh number for cut")/= 0)then
        go to 18
    else
        go to 17
    end if
18  continue
    read(7,*)Nummesh
    
    if(Nummesh/=0)then
    allocate(Ordermesh(Nummesh))
    read(7,*)
    read(7,*)Ordermesh(:)
    sec_p=0
    do i=1,Nummesh
        sec_p=Nij(Ordermesh(i),2)+sec_p
    end do
    !write(*,*)sec_p
    allocate(secx(2*sec_p+1),secx1(2*sec_p+1),secy(2*sec_p+1),secz(2*sec_p+1),seccp(2*sec_p+1))
    allocate(ylift(Numcut,2),spanlift(Numcut,2))
    open(unit=8,FILE="A502/iflggp",action="read",status="old")
    write(9,'(A)')"VARIABLES= x,x/c,y,z,cp"
19   read(8,'(A)')string1
    if (index(string1,"$sectional pressures")/= 0)then
        go to 20
    else
        go to 19
    end if
20  continue
    read(8,*)
    read(8,*)
    !write(*,*)string1
    do i=1,Numcut
        write(9,'(A)')"zone"
        read(8,*)
        !write(*,*)string1
        j=1
        do while(.TRUE.)
            
            read(8,*,err=23)secx(j),secy(j),secz(j),secx1(j),seccp(j)
            write(9,*)secx(j),secx1(j),secy(j),secz(j),seccp(j)
            j=j+1
        end do
23      continue   
        
        
        sec_p=j
        secx(j)=secx(1)
        secx1(j)=secx1(1)
        secy(j)=secy(1)
        secz(j)=secz(1)
        seccp(j)=seccp(1)
        write(9,*)secx(j),secx1(j),secy(j),secz(j),seccp(j)
         !read(8,*)string1
        !write(*,*)string1
        
        
        !!======展向升力分布 2023.2.19
        
            ylift(i,1)=0
            ylift(i,2)=secy(j)
            do j=1,sec_p-1
                !write(*,*)secx(j),seccp(j)
                if (secx(2)>secx(1))then
                ylift(i,1)=ylift(i,1)+(secx(j+1)-secx(j))*(seccp(j)+seccp(j+1))*0.5
                else
                ylift(i,1)=ylift(i,1)-(secx(j+1)-secx(j))*(seccp(j)+seccp(j+1))*0.5    
                
                end if 
                
            end do
            !write(*,*) ylift(i,:)
    end do
    
    
        do j=1,Numcut-1
            spanlift(j,2)=0.5*rou*(Mach*c)**2*(ylift(j,1)+ylift(j+1,1))*(ylift(j+1,2)-ylift(j,2))*0.5
            spanlift(j,1)=ylift(j,2)
            !write(*,*)spanlift(j,1)
        end do
        
        spanlift(Numcut,1)=ylift(Numcut,2)
        spanlift(Numcut,2)=0
       open(unit=11,FILE="A502/span_distribution.dat")
        do j=1,Numcut
            write(11,*)spanlift(j,:)
        end do
    
    end if
    
            
        
!============================================================================================输出机翼剖面信息


!=============================输出气动力系数估算值===
open(unit=12,FILE="A502/areo.dat")
open(unit=13,FILE="A502/ffm")
do i=1,11
    read(13,*)
end do
read(13,*)tem,tem,tem,cl,cdp
read(13,*)tem,cm,tem,Swet

fm=(1+0.15*Mach**2)**(-0.58)
Reno=rou*c*Mach*chord/miu
cfi=0.455/(log10(Reno)**2.58)
cdf=cfi*fm*Swet/Sref
ld=cl/(cdf+cdp)
cd=cdf+cdp
cm=(cm+cl*xref)/chord

write(12,*)"cl  cdp   cdf  cd  ld  cm"
write(12,'(6(F10.5))')cl,cdp,cdf,cd,ld,cm
!=============================输出气动力系数估算值===

!============================================================================================读取输出模型和表面压力系数2020.9.1添加
    write(10,'(A)')"VARIABLES= x,y,z,cp"
21  read(6,'(A)')string1
    if (index(string1,"dupt")/= 0)then
        go to 22
    else
        go to 21
    end if
22  continue
    read(6,*)
    !read(6,*)
    read(6,*)
    !write(*,*)string1
    do i=1,Nmesh
        write(10,*)"zone "," i=",Nij(i,1)+1," j=",Nij(i,2)+1
        !allocate(ox(),oy(),oz(),ocp())
        do j=1,Nij(i,2)+1
            read(6,*)
            do k=1,Nij(i,1)+1
                !write(*,*)"OK"
                read(6,*)ox,ox,oy,oz,ocp
                write(10,*)ox,oy,oz,ocp
            end do
            read(6,*)
        end do
    end do
  
!============================================================================================读取输出模型和表面压力系数 2020.9.1添加

    close(unit=1)
    close(unit=2)
    close(unit=3)
    close(unit=4)
    close(unit=5)
    close(unit=6)
    close(unit=7)
    close(unit=8)
    close(unit=9)
    close(unit=10)
    end subroutine A502out1
    
    
    
    
    subroutine A502out2
    implicit none
    integer i,j,k,ii,jj,kk,Nmesh,Nfile,Nele,Nelei,Nlast,NN,Nnear,Numcut,Sec_P,Nummesh   !Nmesh网格块数，Nfile流场数据读取的段数，Nele总面元数，Nelei单块网格面元数,Nlast最后一段数据的面元数,NN 面元循环累计计数器,Nnear近场点数
    real mach,Sref,AOA,X0,xnear,dpp,Total                         !X0用于坐标转换,dpp是dp/p
    real miu,chord,Reno,fm,cfi,cdf,cdp,cl,ld,Swet,cd,xref,cm
    integer,allocatable::Nij(:,:)              !Nij各块网格的维度数，该数组为Nmesh*2维
    integer,allocatable::Ordermesh(:)  !做剖面信息的网格编号
    real,allocatable:: x(:),y(:),z(:),nx(:),ny(:),nz(:),Cp(:),secx(:),secy(:),secz(:),secx1(:),seccp(:)
    real,allocatable::Lift(:,:),Larea(:,:)!,ox(:,:),oy(:,:),oz(:,:),ocp(:,:)      !lift升力分布，Larea升力面积分布
    real,allocatable::Fz(:,:),Ftran(:,:)        !Fz每个面元的升力,Ftran转换X坐标的升力数组
    real,allocatable::Ylift(:,:),spanlift(:,:) 
    real::H=-30.0                               !用于进行轴向坐标转换的位置
    real rou,p0,B
    real ox,oy,oz,ocp
    real::c=295.07
    real::pi=3.141592658
    real tem,temp                                    !临时数据，用于赋值不需要的数据
!    real int                                    !临时整形数据，用于赋值不需要的数据
    real::gamma=1.4
    integer chazhi
    character(len=256)::string1                
    character(len=10)::string2
    
    !======================================!!子程序A502out变量说明!!!=========================================！    
    !Nmesh,Nfile，Nele                        网格块数，数据的段数，总面元数！
    !Nelei，Nlast,NN,Nnear                    单块网格面元数,最后一段数据的面元数，面元循环累计计数器,近场点数！  
    !mach,Sref,AOA,X0,xnear,dpp,Total         马赫数，半模参考面积，迎角，坐标转换x，近场点数，dp/p，总升力   ！
    !Nij(:,:)                                 Nij各块网格的维度数，该数组为Nmesh*2维                          ！
    !x(:),y(:),z(:),nx(:),ny(:),nz(:),Cp(:)   面元中心点XYZ坐标及其法向量，压力系数                           ！
    !Lift(:,:),Larea(:,:)                     升力分布，升力面积分布                                          !
    !Fz(:,:),Ftran(:,:)                       每个面元的升力,转换X坐标的升力数组                              !
    !chazhi                                   等效截面积插值点数                                              ！
    !======================================!!子程序A502out变量说明!!!=========================================！
    
    open(unit=1,FILE="A502/panair.out",action="read",status="old")
    open(unit=2,FILE="A502/Lift distribution.dat",action="write",status="replace")
    open(unit=3,FILE="A502/Lift area.dat",action="write",status="replace")
    open(unit=4,FILE="indata/FABoom.in",action="read",status="old")
    open(unit=5,FILE="A502/A502 nearfield signal.dat",action="write",status="replace")
    open(unit=6,FILE="A502/agps",action="read",status="old")
    open(unit=7,FILE="indata/geo_information2.in",action="read",status="old")
    !
    open(unit=9,FILE="A502/cut.dat",action="write",status="replace")
    open(unit=10,FILE="A502/surface_Cp.dat",action="write",status="replace")
    !write(*,*)"          000000     0           00000   0000   0000  0     0"
    !write(*,*)"          0         0 0          0    0 0    0 0    0 00   00"
    !Write(*,*)"          000000   0   0  00000  00000  0    0 0    0 0 0 0 0"
    !Write(*,*)"          0       0000000        0    0 0    0 0    0 0  0  0"
    !Write(*,*)"          0      0       0       00000   0000   0000  0     0"

    !write(*,*)"输入密度, 压强,插值点数"
    !read(*,*)rou
    !read(*,*)p0
    !read(*,*)chazhi
!=======================================================================读取inp文件    
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*) 
      
    read(4,*)tem,AOA,tem,rou,p0
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*)Numcut
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*)chazhi
    read(4,*)
    read(4,*)
    read(4,*)
    read(4,*)miu,chord,xref
    
    
    
   
    
!=======================================================================读取马赫数    
1   read(1,'(A)')string1
    if (index(string1,"$mach number")/= 0)then
        go to 2
    else
        go to 1
    end if
2   continue    
    read(1,*)tem,Mach     
    
!========================================================================读取参考面积    
3   read(1,'(A)')string1
    if (index(string1,"$references for accumulated forces and moments")/= 0)then
        go to 4
    else
        go to 3
    end if
4   continue    
    read(1,*)
    read(1,*)tem,Sref  
    
!=========================================================================读取网格块数   
!5   read(1,'(A)')string1
!    if (index(string1,"$points - wing-body  with composite panels")/= 0)then
!        go to 6
!    else
!        go to 5
!    end if
!6   continue    
    read(7,*)
    read(7,*)Nmesh,Nmesh                                 
    !write(*,*)Mach,Sref,Nmesh
  
!=========================================================================读取近场点数
13  read(1,'(A)')string1
    if (index(string1,"$xyz")/= 0)then
        go to 14
    else
        go to 13
    end if
14  continue
    read(1,*)tem,Nnear
    
!=========================================================================读取每块网格面元维数(不是网格维数）   
7   read(1,'(A)')string1
    if (index(string1,"network id&index   #rows   #cols")/= 0)then
        go to 8
    else
        go to 7
    end if
8   continue   
    read(1,*)
    allocate(Nij(Nmesh,2))
    do i=1,Nmesh
        read(1,*)string2,tem,Nij(i,1),Nij(i,2)
        Nij(i,1)=Nij(i,1)-1
        Nij(i,2)=Nij(i,2)-1
        !write(*,*)Nij(i,:)
    end do
   
 !=========================================================================计算总面元数
    Nele=Nij(1,1)*Nij(1,2)
    do i=2,Nmesh
        Nele=Nele+Nij(i,1)*Nij(i,2)
        !write(*,*)Nele
    end do
    !write(*,*)Nele
    
    
!=========================================================================建立读取面元流场信息的数组x,y,z,nx,ny,nz,cp
    allocate(x(Nele),y(Nele),z(Nele),nx(Nele),ny(Nele),nz(Nele),Cp(Nele))
9   read(1,'(A)')string1
    if (index(string1,"simultaneous solution number")/= 0)then
        go to 10
    else
        go to 9
    end if
10  continue 
    !write(*,*)string1
!=========================================================================开始读取流场信息
!=========================================================================开始读取流场信息   
!=========================================================================开始读取流场信息    
    NN=1
     
    do i=1,Nmesh                                   !大循环是网格块数，读取块数次网格
        Nelei=Nij(i,1)*Nij(i,2)                    !求本块网格面元数
        Nfile=int(Nelei/10)
        Nlast=mod(Nelei,10)
       ! write(*,*)Nfile,Nlast
11       read(1,'(A)')string1
        if (index(string1,"0*e*for-mom")/= 0.OR.index(string1,"freestream velocity")/= 0)then  !寻找数据开始特征字段
            go to 12
        else
            go to 11
        end if
12      continue 
        !write(*,*)"牛批"

        do j=1,Nfile                               !循环数据段数
            !write(*,*)"牛批",j
            do ii=1,7
                read(1,*)! string1
            end do
            !write(*,*)string1
            do jj=1,10                        !循环每段下面10段数据
                read(1,*)tem,tem,x(NN),y(NN),z(NN),tem,tem,tem,tem,tem,nx(NN),ny(NN),nz(NN)
                read(1,'(11E11.4)')tem,tem,tem,tem,tem,tem,tem,tem,tem,tem,Cp(NN)
                read(1,*)
                read(1,*)
                !write(*,*)NN,Cp(NN)!NN,x(NN),y(NN),z(NN),nx(NN),ny(NN),nz(NN),
                NN=NN+1
                !write(*,*)i,j,jj
            end do
            !write(*,*)"牛批",j
        end do
        
        if(Nlast>0)then      !对非10倍数的Nelei读取剩下的数据
            do ii=1,7
                read(1,*)
            end do
            do jj=1,Nlast                        !循环余下的Nlast段数据
                read(1,*)tem,tem,x(NN),y(NN),z(NN),tem,tem,tem,tem,tem,nx(NN),ny(NN),nz(NN)
                read(1,'(11E11.4)')tem,tem,tem,tem,tem,tem,tem,tem,tem,tem,Cp(NN)    
                read(1,*)
                read(1,*)
                !write(*,*)NN,x(NN)!,y(NN),z(NN),nx(NN),ny(NN),nz(NN),
                NN=NN+1
            end do
        end if
         !write(*,*)"牛批"
        !write(*,*)NN,"mesh",i,"done"
    end do
    !write(*,*)Cp(Nele)
    !write(*,*)"牛批"
    !write(*,*)"OK"
!=========================================================================流场信息读取完成
!=========================================================================流程信息读取完成   
!=========================================================================流程信息读取完成
        
!=========================================================================计算并赋值面元升力数组
    allocate(Fz(Nele,3))                              !Nele列，3维，第一列x，第二列z，第三列Fz
    !do i=1,Nele
    !    Fz(i,1)=x(i)
    !    Fz(i,2)=z(i)
    !    Fz(i,3)=-Cp(i)*0.5*rou*(Mach*c)**2*nz(i)*Sref
    !end do
    !!write(*,*)Fz(Nele,:)
    !Total=0
    !do i=1,Nele
    !    Total=Fz(i,3)+Total
    !end do
  
!    
!!==========================================================================不坐标转换
!    allocate(Ftran(Nele,2))
!
!    do i=1,Nele
!        Ftran(i,1)=Fz(i,1)   
!        Ftran(i,2)=Fz(i,3)
!        !write(*,*)Ftran(i,1)
!    end do

!!=========================================================================坐标转换  
!allocate(Fz(Nele,3))                              !Nele列，3维，第一列x，第二列z，第三列Fz
    do i=1,Nele
        Fz(i,1)=cos(AOA*pi/180)*x(i)+sin((AOA*pi/180))*z(i)          !旋转网格
        Fz(i,2)=-sin(AOA*pi/180)*x(i)+cos((AOA*pi/180))*z(i)
        Fz(i,3)=-Cp(i)*0.5*rou*(Mach*c)**2*nz(i)*Sref
    end do
    !write(*,*)Fz(Nele,:)
    Total=0
    do i=1,Nele
        Total=Fz(i,3)+Total
    end do
    write(*,*)"重量为",total*0.2,"kg" 
    
    allocate(Ftran(Nele,2))
    do i=1,Nele
        
        Ftran(i,1)=Fz(i,1)+(Fz(i,2)-H)*(Mach**2-1)**0.5    
        Ftran(i,2)=Fz(i,3)
        
    end do
!=========================================================================对X排序升力数组  
    do i=1,Nele-1
        do j=i+1,Nele
            if(Ftran(j,1)<Ftran(i,1))then
                tem=Ftran(i,1)
                Ftran(i,1)=Ftran(j,1)
                Ftran(j,1)=tem
                tem=Ftran(i,2)
                Ftran(i,2)=Ftran(j,2)
                Ftran(j,2)=tem
            end if
        end do
    end do
    !do i=1,Nele
        !write(*,*)Fz(1,1)! Ftran(1,1)!,Ftran(i,1)
    !end do
!=========================================================================升力分布求和
    allocate(Lift(Nele,2))
    tem=Ftran(1,1)
    Lift(1,1)=0!Ftran(1,1)! 2020 9.7  Fz(1,1)
    Lift(1,2)=Ftran(1,2)
   ! write(*,*)lift(1,1),lift(1,2)
    do i=2,Nele
        Lift(i,1)=Ftran(i,1)-tem!+Fz(1,1)!(Fz(1,2)-H)*(Mach**2-1)**0.5!
        Lift(i,2)=Lift(i-1,2)+Ftran(i,2)
        !write(2,*)Lift(i,1),Lift(i,2)
    end do
    !write(*,*)Lift(Nele,2)*2
!=========================================================================升力分布插值    
    allocate(Larea(chazhi,2))
    do i=1,chazhi
        Larea(i,1)=(Lift(Nele,1)-Lift(1,1))/(chazhi-1)*(i-1)+Lift(1,1)
    end do
    Larea(1,2)=Lift(1,2)*2
    Larea(chazhi,2)=Lift(Nele,2)*2
    write(2,*)Larea(1,1),Larea(1,2)
    do i=2,chazhi-1
        do j=1,Nele-1
            if(Larea(i,1)>Lift(j,1).AND.Larea(i,1)<=Lift(j+1,1))then
                Larea(i,2)=Lift(j,2)+(Lift(j+1,2)-Lift(j,2))/(Lift(j+1,1)-Lift(j,1))*(Larea(i,1)-Lift(j,1))
            end if
        end do
        Larea(i,2)=Larea(i,2)*2
        write(2,*)Larea(i,1),Larea(i,2)
    end do
    write(2,*)Larea(chazhi,1),Larea(chazhi,2)
    !write(*,*)Larea(i,1),Larea(chazhi,2)
!=========================================================================计算升力等效截面积分布    
    B=(Mach**2-1)**0.5
    write(3,*)chazhi+2!"VARIABLES=X,S<sub>L</sub>(x)"
    Write(3,*)0,0
    do i=1,chazhi
        Larea(i,2)=Larea(i,2)*B/(rou*(Mach*c)**2)!(gamma*p0*Mach**2)
        write(3,*)Larea(i,:)
    end do
    write(3,*)Larea(chazhi,1)*1.2,Larea(chazhi,2)    !!!2020.11.4改长度
    
    
!============================================================================================读取输出dp/p    
15   read(1,'(A)')string1
    if (index(string1,"off body flow characteristics")/= 0)then
        go to 16
    else
        go to 15
    end if
16  continue
    !write(*,*)"niupi"
    read(1,*)
    read(1,*)
    read(1,*)
    write(5,*)"VARIABLES=X,dp/p"
    do i=1,Nnear
        read(1,*)tem,tem,xnear,tem,tem,tem,tem,tem,tem,dpp
        dpp=dpp*0.5*rou*(Mach*c)**2/p0
        write(5,*)xnear,dpp
    end do
    
    
    



!============================================================================================输出机翼剖面信息 2020.9.2修改了输出剖面的bug
17   read(7,'(A)')string1
    if (index(string1,"mesh number for cut")/= 0)then
        go to 18
    else
        go to 17
    end if
18  continue
    read(7,*)Nummesh
    
    if(Nummesh/=0)then
    allocate(Ordermesh(Nummesh))
    read(7,*)
    read(7,*)Ordermesh(:)
    sec_p=0
    do i=1,Nummesh
        sec_p=Nij(Ordermesh(i),2)+sec_p
    end do
    !write(*,*)sec_p
    allocate(secx(2*sec_p+1),secx1(2*sec_p+1),secy(2*sec_p+1),secz(2*sec_p+1),seccp(2*sec_p+1))
    allocate(ylift(Numcut,2),spanlift(Numcut,2))
    open(unit=8,FILE="A502/iflggp",action="read",status="old")
    write(9,'(A)')"VARIABLES= x,x/c,y,z,cp"
19   read(8,'(A)')string1
    if (index(string1,"$sectional pressures")/= 0)then
        go to 20
    else
        go to 19
    end if
20  continue
    read(8,*)
    read(8,*)
    !write(*,*)string1
    do i=1,Numcut
        write(9,'(A)')"zone"
        read(8,*)
        !write(*,*)string1
        j=1
        do while(.TRUE.)
            
            read(8,*,err=23)secx(j),secy(j),secz(j),secx1(j),seccp(j)
            write(9,*)secx(j),secx1(j),secy(j),secz(j),seccp(j)
            j=j+1
        end do
23      continue 
        !write(*,*)j
        sec_p=j
        secx(j)=secx(1)
        secx1(j)=secx1(1)
        secy(j)=secy(1)
        secz(j)=secz(1)
        seccp(j)=seccp(1)
        write(9,*)secx(j),secx1(j),secy(j),secz(j),seccp(j)
         !read(8,*)string1
        !write(*,*)string1
        
        
        !!======展向升力分布 2023.2.19
        
            ylift(i,1)=0
            ylift(i,2)=secy(j)
            do j=1,sec_p-1
                !write(*,*)secx(j),seccp(j)
                if (secx(2)>secx(1))then
                ylift(i,1)=ylift(i,1)+(secx(j+1)-secx(j))*(seccp(j)+seccp(j+1))*0.5
                else
                ylift(i,1)=ylift(i,1)-(secx(j+1)-secx(j))*(seccp(j)+seccp(j+1))*0.5    
                
                end if 
                
            end do
            !write(*,*) ylift(i,:)
    end do
    
    
        do j=1,Numcut-1
            spanlift(j,2)=0.5*rou*(Mach*c)**2*(ylift(j,1)+ylift(j+1,1))*(ylift(j+1,2)-ylift(j,2))*0.5
            spanlift(j,1)=ylift(j,2)
            !write(*,*)spanlift(j,1)
        end do
        
        spanlift(Numcut,1)=ylift(Numcut,2)
        spanlift(Numcut,2)=0
       open(unit=11,FILE="A502/span_distribution.dat")
        do j=1,Numcut
            write(11,*)spanlift(j,:)
        end do
        
        
    
    
    end if
    
            
        
!============================================================================================输出机翼剖面信息

!=============================输出气动力系数估算值===
open(unit=12,FILE="A502/areo.dat")
open(unit=13,FILE="A502/ffm")
do i=1,11
    read(13,*)
end do
read(13,*)tem,tem,tem,cl,cdp
read(13,*)tem,cm,tem,Swet

fm=(1+0.15*Mach**2)**(-0.58)
Reno=rou*c*Mach*chord/miu
cfi=0.455/(log10(Reno)**2.58)
cdf=cfi*fm*Swet/Sref
ld=cl/(cdf+cdp)
cd=cdf+cdp
cm=(cm+cl*xref)/chord

write(12,*)"cl  cdp   cdf  cd  ld  cm"
write(12,'(6(F10.5))')cl,cdp,cdf,cd,ld,cm
!=============================输出气动力系数估算值===

                

!============================================================================================读取输出模型和表面压力系数2020.9.1添加
    write(10,'(A)')"VARIABLES= x,y,z,cp"
21  read(6,'(A)')string1
    if (index(string1,"dupt")/= 0)then
        go to 22
    else
        go to 21
    end if
22  continue
    read(6,*)
    !read(6,*)
    read(6,*)
    !write(*,*)string1
    do i=1,Nmesh
        write(10,*)"zone "," i=",Nij(i,1)+1," j=",Nij(i,2)+1
        !allocate(ox(),oy(),oz(),ocp())
        do j=1,Nij(i,2)+1
            read(6,*)
            do k=1,Nij(i,1)+1
                !write(*,*)"OK"
                read(6,*)ox,ox,oy,oz,ocp
                write(10,*)ox,oy,oz,ocp
            end do
            read(6,*)
        end do
    end do
  
!============================================================================================读取输出模型和表面压力系数 2020.9.1添加

    close(unit=1)
    close(unit=2)
    close(unit=3)
    close(unit=4)
    close(unit=5)
    close(unit=6)
    close(unit=7)
    close(unit=8)
    close(unit=9)
    close(unit=10)
    close(unit=11)
    end subroutine A502out2

    !===================================================================================================！
    !                                                                                                   ！
    !                                     体积等效截面积程序slice                                       ！
    !                                     体积等效截面积程序slice                                       ！
    !                                     体积等效截面积程序slice                                       ！
    !                                      2020.7.26完成子程序V1                                        ！    
    !===================================================================================================！     
    
    subroutine slice
    implicit none
    character(len=256)::str1
    character(len=8)::str2
    real(kind=8)::H=-30    !等效截面积x序列的参考线，在飞机下方的距离
    real::pi=3.141592658
    real(kind=8) mach,AOA,L,LM,AH,S,lamda,temp
    integer Nmesh,Nrow,Ncolumn,Np,i,j,k,N,Nslice,deltax,id,startslice,Xstart,Xend
    real(kind=8),allocatable::x(:,:),y(:,:),z(:,:),xt(:,:),yt(:,:),zt(:,:),xyzcross(:,:),xty(:,:)
    real(kind=8),allocatable::xslice(:),Sslice(:)
    open(unit=1,file="tempdata/geometry.dat",action='read',status='old')
    open(unit=2,file="indata/FABoom.in",action='read',status='old')
    open(unit=3,file="slice/Volume area.dat",action="write",status="replace")
    open(unit=4,file="slice/rotate.dat",action="write",status="replace")
    
    !=================读取indata文件中的参数=============!
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*)mach,AOA
    read(2,*)
    read(2,*)L,LM,AH,S
    !write(*,*)mach,AOA,L,LM,S
    
    
    !=================读取dat文件中的几何参数并记录网格块数=============!
    Nmesh=0
    do while(.true.)
        read(1,*,END=1)str1
           if(index(str1,"zone")/=0)then
           Nmesh=Nmesh+1
        end if
    end do
1   continue
    !write(*,*)Nmesh
    rewind(1)
    
    !=================建立马赫切面的x、z坐标=============!
    Nslice=150
    allocate(xslice(Nslice),sslice(Nslice))
    deltax=AH*(mach**2-1)**0.5
    do i=1,Nslice
        xslice(i)=-0.15*L+(L+0.5*L+deltax)/(Nslice-1)*(i-1)-H*(mach**2-1)**0.5    !!!!!!  2020.11.4改长度
       ! write(*,*)xslice(i)
    end do
    
    sslice=0
        
        
    
    !================开始网格块数循环===============!
    !================开始网格块数循环===============!
    !================开始网格块数循环===============!
    
    !==以每块网格为一次操作，每个大循环下，读取一个网格后，完成xslice的遍历===!
    !=====求得一次sslice，下一次循环对sslice进行累加==!
    do k=1,Nmesh          !大循环
        read(1,*)str2,str2,Nrow,str2,Ncolumn !读取行数和列数
        allocate(x(Nrow,Ncolumn),y(Nrow,Ncolumn),z(Nrow,Ncolumn))   !这样定义是按一列一列的填充矩阵
        allocate(xt(Nrow,Ncolumn),yt(Nrow,Ncolumn),zt(Nrow,Ncolumn))
        Np=Nrow*Ncolumn
        
    !================读取目前网格的坐标点===============!    
        do j=1,Ncolumn
            do i=1,Nrow
                read(1,*)x(i,j),y(i,j),z(i,j)
                !write(*,*)x(i,j),y(i,j),z(i,j)
            end do
        end do
        
    !================对目前网格的坐标点进行迎角旋转===============!  
        write(4,*)"zone "," i=",Nrow," j=",Ncolumn
        do j=1,Ncolumn
            do i=1,Nrow
                xt(i,j)=cos(AOA*pi/180)*x(i,j)+sin((AOA*pi/180))*z(i,j)
                yt(i,j)=y(i,j)
                zt(i,j)=-sin(AOA*pi/180)*x(i,j)+cos((AOA*pi/180))*z(i,j)
                write(4,*)xt(i,j),yt(i,j),zt(i,j)
            end do
        end do
   !!================不旋转===============!  
   !     write(4,*)"zone "," i=",Nrow," j=",Ncolumn
   !     do j=1,Ncolumn
   !         do i=1,Nrow
   !             xt(i,j)=x(i,j)
   !             yt(i,j)=y(i,j)
   !             zt(i,j)=z(i,j)
   !             write(4,*)xt(i,j),yt(i,j),zt(i,j)
   !         end do
   !     end do  
        
        
        
     !================对旋转后的网格开始遍历xslice,查找交点===============!    
     !声明一个维数为网格宽度的二维数组存放交点坐标
     !声明一个二维数组存放将旋转后的几何网格点按马赫线投影到H位置的x坐标
       allocate(xyzcross(Nrow,3),xty(Nrow,Ncolumn))      !声明一个维数为网格宽度的二维数组存放交点坐标
       !xyzcross=0
       
     
     !================对所有网格点投影到H高度的位置，求x坐标===============! 
        do j=1,Ncolumn
            do i=1,Nrow
                xty(i,j)=xt(i,j)+(zt(i,j)-H)*(mach**2-1)**0.5
                !write(*,*)xty(i,j)
            end do
        end do
     !!================垂直投影===============! 
     !   do j=1,Ncolumn
     !       do i=1,Nrow
     !           xty(i,j)=xt(i,j)
     !           !write(*,*)xty(i,j)
     !       end do
     !   end do   
        
        
     !================遍历xslice坐标，查找每个slice上的交点，记录坐标位置===============!
        do n=1,Nslice
            id=0  !找到的交点书
            xyzcross(:,:)=0
            do i=1,Nrow-1
                
                do j=1,Ncolumn-1                   !==================!查找第i行的线段是否有交点
                    if((xty(i,j)<=xslice(n).AND.xty(i,j+1)>xslice(n)).OR.(xty(i,j)>=xslice(n).AND.xty(i,j+1)<xslice(n)))then   !判断条件有bug
                        id=id+1
                        !write(*,*)xslice(n)
                        lamda=(xslice(n)-xty(i,j))/(xty(i,j+1)-xty(i,j))
                        xyzcross(id,1)=xt(i,j)+lamda*(xt(i,j+1)-xt(i,j))
                        !call spline(xt(i,:),yt(i,:),Ncolumn,xyzcross(id,1),xyzcross(id,2),j)       ! if use the spline interpolation, please cancel the comment form.
                        !call spline(xt(i,:),zt(i,:),Ncolumn,xyzcross(id,1),xyzcross(id,3),j)
                        xyzcross(id,2)=yt(i,j)+lamda*(yt(i,j+1)-yt(i,j))
                        xyzcross(id,3)=zt(i,j)+lamda*(zt(i,j+1)-zt(i,j))
                    end if
                end do                              !==================!查找第i行的线段是否有交点
                
                if(xty(i,Ncolumn)==xslice(n))then   !==================!查找第i行的最后一个点是否是交点
                    id=id+1
                    xyzcross(id,1)=xt(i,Ncolumn)
                    xyzcross(id,2)=yt(i,Ncolumn)
                    xyzcross(id,3)=zt(i,Ncolumn)
                end if
                
                if((xty(i,1)<xslice(n).AND.xty(i+1,1)>xslice(n)).OR.(xty(i,1)>xslice(n).AND.xty(i+1,1)<xslice(n)))then  !查找前缘交点
                    id=id+1
                    lamda=(xslice(n)-xty(i,1))/(xty(i+1,1)-xty(i,1))
                    xyzcross(id,1)=xt(i,1)+lamda*(xt(i+1,1)-xt(i,1))
                    xyzcross(id,2)=yt(i,1)+lamda*(yt(i+1,1)-yt(i,1))
                    xyzcross(id,3)=zt(i,1)+lamda*(zt(i+1,1)-zt(i,1))
                end if
                
                if((xty(i,Ncolumn)<xslice(n).AND.xty(i+1,Ncolumn)>xslice(n)).OR.(xty(i,Ncolumn)>xslice(n).AND.xty(i+1,Ncolumn)<xslice(n)))then  !查找后缘交点
                    id=id+1
                    lamda=(xslice(n)-xty(i,Ncolumn))/(xty(i+1,Ncolumn)-xty(i,Ncolumn))
                    xyzcross(id,1)=xt(i,Ncolumn)+lamda*(xt(i+1,Ncolumn)-xt(i,Ncolumn))
                    xyzcross(id,2)=yt(i,Ncolumn)+lamda*(yt(i+1,Ncolumn)-yt(i,Ncolumn))
                    xyzcross(id,3)=zt(i,Ncolumn)+lamda*(zt(i+1,Ncolumn)-zt(i,Ncolumn))
                end if
                   
            end do
            
            !=======还有第Nrow行未查找======！
            !=======开始查找第Nrow行======！
             do j=1,Ncolumn-1                   !==================!查找第Nrow行的线段是否有交点
                if((xty(Nrow,j)<=xslice(n).AND.xty(Nrow,j+1)>xslice(n)).OR.(xty(Nrow,j)>=xslice(n).AND.xty(Nrow,j+1)<xslice(n)))then   !判断条件有bug
                        id=id+1
                        !write(*,*)xslice(n)
                        lamda=(xslice(n)-xty(Nrow,j))/(xty(Nrow,j+1)-xty(Nrow,j))
                        xyzcross(id,1)=xt(Nrow,j)+lamda*(xt(Nrow,j+1)-xt(Nrow,j))
                        xyzcross(id,2)=yt(Nrow,j)+lamda*(yt(Nrow,j+1)-yt(Nrow,j))
                        xyzcross(id,3)=zt(Nrow,j)+lamda*(zt(Nrow,j+1)-zt(Nrow,j))
                end if
             end do                              !==================!查找第i行的线段是否有交点
             
             !==================!查找(Nrow,Ncolumn)是否交点
            if(xty(Nrow,Ncolumn)==xslice(n))then
                id=id+1
                id=id+1
                xyzcross(id,1)=xt(Nrow,Ncolumn)
                xyzcross(id,2)=yt(Nrow,Ncolumn)
                xyzcross(id,3)=zt(Nrow,Ncolumn)
            end if
            
            !if(id/=0)then
            !write(*,*)id,xyzcross(id,2),xslice(n)
            !end if
            
            
        !==================截面积计算部分=================！
        !==================截面积计算部分=================！
        !==================截面积计算部分=================！
        !==================开始求当前xyzcross的坐标点的面积，对y和z坐标进行梯形积分==============！
           ! if(id/=0)then
                do i=1,id-1               !!!Nrow-1  !之前的写法会导致id个点和后面的0求面积，就不正确了
                    Sslice(n)=(xyzcross(i,2)+xyzcross(i+1,2))*0.5*(xyzcross(i,3)-xyzcross(i+1,3))+Sslice(n)
                end do
           
          !  end if

            
            
            
        end do          !到这里xslice遍历完了
  
        
        
        
        
        deallocate(x,y,z,xt,yt,zt,xty,xyzcross)
    end do
    
   ! write(3,*)"VARIABLES=X,S<sub>V</sub>(x)"
    do n=1,nslice
        xslice(n)=xslice(n)+H*(mach**2-1)**0.5
        sslice(n)=2*sslice(n)
        !write(3,*)xslice(n),sslice(n)
    end do
    
    Xstart = 2
    do n=2,nslice-2   !去除非零的头尾
        if(sslice(n)==0.AND.sslice(n+1)/=0.AND.sslice(n-1)==0)then
            Xstart=n
            !write(*,*)Xstart
            goto 2
        end if
    end do
2   continue    
    
    Xend = nslice
    do n=2,nslice-2 
        if(sslice(n)==0.AND.sslice(n+1)==0.AND.sslice(n-1)/=0)then
            Xend=n
            !write(*,*)Xend
            goto 3
        end if
    end do
3   continue  

!    Xend = nslice     !==============================================2022.4.28改从末端出发剪去0值数据
!    do n=nslice-2,2 
!         if(sslice(n)==0.AND.sslice(n+1)==0.AND.sslice(n-1)/=0)then
!            Xend=n
!            write(*,*)Xend
!            goto 3
!        end if
!    end do
!3   continue

    
    write(3,*)Xend-Xstart+1
    temp=xslice(Xstart)
    do n=Xstart,Xend
        xslice(n)= xslice(n)-temp
        write(3,*)xslice(n),sslice(n)
    end do
    
    
    end subroutine slice
    
    
        
    
    
    
        
    
    
    !========================================================================================================！
    !                                                                                                        ！
    !                                  等效截面积计算程序area_distribution                                   ！
    !                                  等效截面积计算程序area_distribution                                   ！
    !                                  等效截面积计算程序area_distribution                                   ！    
    !                                                                                                        ！
    !========================================================================================================！
subroutine area_distribution2
    implicit none
    integer i,j,k,VNum,LNum,Num,cha
    real dx
    real,allocatable::xv(:),Sv(:),xl(:),Sl(:)
    real,allocatable::Vcha(:,:),Lcha(:,:),Sx(:,:)
    open(unit=1,file="slice/Volume area.dat",action="read",status="old")
    open(unit=2,file="A502/Lift area.dat",action="read",status="old")
    open(unit=3,file="area/Area_due_to_Volume.dat",action="write",status="replace")
    open(unit=4,file="area/Area_due_to_Lift.dat",action="write",status="replace")
    open(unit=5,file="sonicboom/sonicboom area.dat",action="write",status="replace")
    open(unit=6,file="area/Total_equivalent_area.dat",action="write",status="replace")  
    
!===========================================================读体积文件
    read(1,*)VNum  !点数
    allocate(xv(VNum),Sv(VNum))
    do i=1,VNum
        read(1,*)xv(i),Sv(i)
        !write(*,*)xv(i),Sv(i)
    end do
 !===========================================================读升力文件
    read(2,*)LNum   !点数
    allocate(xl(LNum),Sl(LNum))
    do i=1,LNum
        read(2,*)xl(i),Sl(i)
        !write(*,*)xl(i),Sl(i)
    end do   
 !===========================================================插值100点
    cha=200
    allocate(Vcha(cha,2),Lcha(cha,2))
    Lcha(1,1)=xl(1)
    Vcha(1,1)=xl(1)
    do i=1,cha
        Lcha(i,1)=Lcha(1,1)+(xv(VNum)-Lcha(1,1))/(cha-1)*(i-1)    !体积分布短一些，用体积分布最后一位数作末端
        !Vcha(i,1)=Lcha(1,1)+(Sv(VNum)-Vcha(1,1))/999*(i-1)
        Vcha(i,1)=Lcha(i,1)
        !write(*,*)Vcha(i,1)
    end do
    Lcha(1,2)=Sl(1)
    Vcha(1,2)=Sl(1)
    do i=2,cha
        do j=1,VNum-1
            if(Vcha(i,1)>xv(j).AND.Vcha(i,1)<=xv(j+1))then
                Vcha(i,2)=Sv(j)+(Sv(j+1)-Sv(j))/(xv(j+1)-xv(j))*(Vcha(i,1)-xv(j))
            end if
        end do 
    end do
    
    do i=2,cha
        do j=1,LNum-1
            if(Lcha(i,1)>xl(j).AND.Lcha(i,1)<=xl(j+1))then
                Lcha(i,2)=Sl(j)+(Sl(j+1)-Sl(j))/(xl(j+1)-xl(j))*(Lcha(i,1)-xl(j))
            end if
        end do 
        !write(*,*)Lcha(i,1)
    end do
 !===========================================================计算总面积分布   
    allocate(Sx(cha,2))
    write(3,*)"VARIABLES=X,S<sub>V</sub>(x)"
    write(4,*)"VARIABLES=X,S<sub>L</sub>(x)"
    write(6,*)"VARIABLES=X,S<sub>X</sub>(x)"
    write(5,*)cha+1
    do i=1,cha
        Sx(i,1)=Lcha(i,1)
        Sx(i,2)=Lcha(i,2)+Vcha(i,2)
        write(5,*)Sx(i,:)
        write(6,*)Sx(i,:)
        write(4,*)Lcha(i,:)
        write(3,*)Vcha(i,:)
    end do
    
    dx=Sx(cha,1)-Sx(cha-1,1)
    !write(5,*)Sx(cha,1)+0.1*,Sx(cha,2)
    !write(6,*)Sx(cha,1)+0.1,Sx(cha,2)
    !write(4,*)Lcha(cha,1)+0.1,Lcha(cha,2)
    write(5,*)Sx(cha,1)+dx,Sx(cha,2)
    write(6,*)Sx(cha,1)*1.05,Sx(cha,2)
    write(4,*)Lcha(cha,1)*1.05,Lcha(cha,2)    
    close(unit=1)
    close(unit=2)
    close(unit=3)
    close(unit=4)
    close(unit=5)
    close(unit=6)
    
    end subroutine area_distribution2 
    
    
    
    subroutine area_distribution1
    implicit none
    integer i,j,k,VNum,LNum,Num,cha
    real dx
    real,allocatable::xv(:),Sv(:),xl(:),Sl(:)
    real,allocatable::Vcha(:,:),Lcha(:,:),Sx(:,:)
    open(unit=1,file="slice/Volume area.dat",action="read",status="old")
    !open(unit=2,file="A502/Lift area.dat",action="read",status="old")
    open(unit=3,file="area/Area_due_to_Volume.dat",action="write",status="replace")
    !open(unit=4,file="area/Area due to Lift.dat",action="write",status="replace")
    open(unit=5,file="sonicboom/sonicboom area.dat",action="write",status="replace")
    open(unit=6,file="area/Total_equivalent_area.dat",action="write",status="replace")  
    
!===========================================================读体积文件
    read(1,*)VNum  !点数
    allocate(xv(VNum),Sv(VNum))
    write(3,*)"VARIABLES=X,S<sub>V</sub>(x)"
    write(6,*)"VARIABLES=X,S<sub>X</sub>(x)"
     write(5,*)VNUM+1
    do i=1,VNum
        read(1,*)xv(i),Sv(i)
        write(3,*)xv(i),Sv(i)
        write(5,*)xv(i),Sv(i)
        write(6,*)xv(i),Sv(i)
    end do
    write(5,*)xv(VNUM)*1.05,Sv(VNUM)
 !!===========================================================读升力文件
 !   read(2,*)LNum   !点数
 !   allocate(xl(LNum),Sl(LNum))
 !   do i=1,LNum
 !       read(2,*)xl(i),Sl(i)
 !       !write(*,*)xl(i),Sl(i)
 !   end do   
 !!===========================================================插值100点
 !   cha=200
 !   allocate(Vcha(cha,2),Lcha(cha,2))
 !   Lcha(1,1)=xl(1)
 !   Vcha(1,1)=xl(1)
 !   do i=1,cha
 !       Lcha(i,1)=Lcha(1,1)+(xv(VNum)-Lcha(1,1))/(cha-1)*(i-1)    !体积分布短一些，用体积分布最后一位数作末端
 !       !Vcha(i,1)=Lcha(1,1)+(Sv(VNum)-Vcha(1,1))/999*(i-1)
 !       Vcha(i,1)=Lcha(i,1)
 !       !write(*,*)Vcha(i,1)
 !   end do
 !   Lcha(1,2)=Sl(1)
 !   Vcha(1,2)=Sl(1)
 !   do i=2,cha
 !       do j=1,VNum-1
 !           if(Vcha(i,1)>xv(j).AND.Vcha(i,1)<=xv(j+1))then
 !               Vcha(i,2)=Sv(j)+(Sv(j+1)-Sv(j))/(xv(j+1)-xv(j))*(Vcha(i,1)-xv(j))
 !           end if
 !       end do 
 !   end do
 !   
 !   do i=2,cha
 !       do j=1,LNum-1
 !           if(Lcha(i,1)>xl(j).AND.Lcha(i,1)<=xl(j+1))then
 !               Lcha(i,2)=Sl(j)+(Sl(j+1)-Sl(j))/(xl(j+1)-xl(j))*(Lcha(i,1)-xl(j))
 !           end if
 !       end do 
 !       !write(*,*)Lcha(i,1)
 !   end do
 !===========================================================计算总面积分布   
    !allocate(Sx(cha,2))
    !write(3,*)"VARIABLES=X,S<sub>V</sub>(x)"
    !write(4,*)"VARIABLES=X,S<sub>L</sub>(x)"
    !write(6,*)"VARIABLES=X,S<sub>X</sub>(x)"
    !write(5,*)cha+1
    !do i=1,cha
    !    Sx(i,1)=Lcha(i,1)
    !    Sx(i,2)=Lcha(i,2)+Vcha(i,2)
    !    write(5,*)Sx(i,:)
    !    write(6,*)Sx(i,:)
    !    write(3,*)Vcha(i,:)
    !end do
    !
    !dx=Sx(cha,1)-Sx(cha-1,1)
    !write(5,*)Sx(cha,1)+0.1*,Sx(cha,2)
    !write(6,*)Sx(cha,1)+0.1,Sx(cha,2)
    !write(4,*)Lcha(cha,1)+0.1,Lcha(cha,2)
    !write(5,*)Sx(cha,1)+dx,Sx(cha,2)
    !write(6,*)Sx(cha,1)*1.05,Sx(cha,2)
    !write(4,*)Lcha(cha,1)*1.05,Lcha(cha,2)    
    close(unit=1)
    !close(unit=2)
    close(unit=3)
    !close(unit=4)
    close(unit=5)
    close(unit=6)
    
    end subroutine area_distribution1
    
    
    
    !========================================================================================================！
    !                                                                                                        ！
    !                                  面积分布2阶导数计算程序diferrential                                   ！
    !                                  面积分布2阶导数计算程序diferrential                                   ！
    !                                  面积分布2阶导数计算程序diferrential                                   ！    
    !                                              2020.6.19                                                 ！
    !========================================================================================================！
    subroutine diferrential
    implicit none
    !implicit double precision (a-h,o-z)
    integer NUM,i
    real,allocatable::x(:),y(:),yy(:),dx(:),dy(:),dyy(:),dy1(:),dy2(:)
    open(unit=1,file="sonicboom/sonicboom area.dat",action="read",status="old")
    open(unit=2,file="sonicboom/2nd_differancial.dat",action="write",status="replace")
    read(1,*) NUM
    
    Num=Num+1
    allocate(x(NUM),y(NUM),yy(NUM),dx(NUM),dy(NUM),dyy(NUM),dy1(NUM),dy2(NUM))
    do i=2,NUM
        read(1,*) x(i),y(i)
        !write(*,*) x(i),y(i)
    end do
    x(1)=x(2)-x(3)
    y(1)=0
   ! write(*,*) x(:)
    ! ======================================================2阶导 2020.8.28修改为中心差分
    do i=2,NUM-1
        dy2(i)=(y(i+1)+y(i-1)-2*y(i))/((x(i+1)-x(i))**2)  !!!
    end do
    dy2(1)=0
    
    dy2(NUM)=0
        
    do i=1,NUM
        write(2,*) x(i),dy2(i)
    end do
    
    !do i=1,NUM
    !    if(i/=NUM)then
    !        dx(i)=x(i+1)-x(i)
    !        dy(i)=y(i+1)-y(i)
    !        dy1(i)=dy(i)/dx(i)
    !    else 
    !        dy1(i)=dy1(i-1)
    !    end if
    !    
    !    !dy1(1)=dy1(2)
    !    !dy1(NUM)=dy1(NUM-1)
    !    !if(i==1)then
    !    !dx(i)=(x(i+1)-x(i))*2
    !    !dy(i)=y(i+1)-y(i)
    !    !dy1(i)=dy(i)/dx(i)
    !    !end if
    !    !if(i==NUM)then
    !    !dx(i)=(x(i)-x(i-1))*2
    !    !dy(i)=y(i)-y(i-1)
    !    !dy1(i)=dy(i)/dx(i)
    !    !end if
    !    
    !    
    !    !write(2,*) x(i),dy1(i)
    !    
    !end do
    
        !write(*,*) x(i),dy1(i)
 ! ======================================================1阶导    
 ! ======================================================2阶导 2020.8.28修改为中心差分
    !do i=1,NUM
    !    if(i/=NUM)then
    !        dyy(i)=dy1(i+1)-dy1(i)
    !        dy2(i)=dyy(i)/dx(i)
    !        else 
    !        dy2(i)=dy2(i-1)
    !    end if
    !    !if(i==1)then
    !    !    dyy(i)=dy1(i+1)-dy1(i)
    !    !    dy2(i)=dyy(i)/dx(i)
    !    !end if
    !    !if(i==NUM)then
    !    !    dyy(i)=dy1(i)-dy1(i-1)
    !    !    dy2(i)=dyy(i)/dx(i)
    !    !end if
    !    !dy2(1)=dy2(2)
    !    !dy2(NUM)=dy2(NUM-1)
    !    write(2,*) x(i),dy2(i)
    !end do
    close(unit=1)
    close(unit=2)
   

    end subroutine diferrential

    !========================================================================================================！
    !                                                                                                        ！
    !                                        F函数计算程序F_function                                         ！
    !                                        F函数计算程序F_function                                         ！
    !                                        F函数计算程序F_function                                         ！    
    !                                              2020.6.19                                                 ！
    !========================================================================================================！
    subroutine F_function
    implicit none
    real,allocatable::x(:),S(:),F(:),y(:),F_hanshu(:),delta_P(:),x1(:),Sum_F(:)
    real pi,gamma,M,B,r,p0,k,DP,XL,XT,root,L,HL,KG
    real,allocatable::Fw(:),fc(:),xc(:),xw(:)
    integer i,j,NUM,root_i,Nc,Nw,kk
    parameter(pi=3.141592653589793)
    parameter(gamma=1.4)
    open(unit=1,file="sonicboom/2nd_differancial.dat",action="read",status="old")
    open(unit=2,file="sonicboom/F_function.dat",action="write",status="replace")
    open(unit=4,file="sonicboom/root.dat",action="write",status="replace")
    open(unit=5,file="sonicboom/delta_P.dat",action="write",status="replace")
    open(unit=6,file="indata/FABoom.in",action="read",status="old")
    read(6,*)
    read(6,*)
    read(6,*)
    read(6,*)
    read(6,*)M,p0,p0,p0,p0,HL
    read(6,*)
    read(6,*)L,L
    r=L*HL
    !M=1.26
    !p0=101000
    !r=20
    B=sqrt(M*M-1)
    
    k=1.0/sqrt(2.0)*(gamma+1)*M**4*B**(-3.0/2.0)
    
!==================================读取面积导数行数
    i=0
    do while(.true.)
    read(1,*,end=10) 
         i=i+1
    end do
10  continue  
    
    close(unit=1)
    NUM=i
    allocate(x(NUM),S(NUM))
!==================================读取面积导数    
    open(unit=3,file="sonicboom/2nd_differancial.dat",action="read",status="old")
    rewind(3)
    
    do i=1,NUM
        read(3,*) x(i),S(i)
    end do 
!==================================计算F-函数   
    Nc=10
    Nw=50

    allocate(F(NUM),y(NUM+Nw),F_hanshu(NUM+Nw))
    do j=1,NUM
        y(j)=x(j)
        F(1)=0
        F(2)=S(2)*(x(2)-x(1))/sqrt(x(2)-x(1))
        !F(NUM-1)=0
        !F(NUM)=0
        if(j>2)then
        do i=2,j-1 !2020.8.28修改f函数末端
            !F(i)=F(i-1)+(x(i)-x(i-1))*S(i)/sqrt(y(j)-x(i))
            F(i)=F(i-1)+(x(i)-x(i-1))*(S(i)/sqrt(y(j)-x(i))+S(i-1)/sqrt(y(j)-x(i-1)))*0.5
            
        end do
        !F(j)=2*F(i-1)-F(i-2)
        F(j)=2*F(j-1)-F(j-2)
        end if
        
        F_hanshu(j)=1.0/(2*pi)*F(j)
        write(2,*)y(j),F_hanshu(j)
    end do
    !F_hanshu(NUM-1)=0
    !F_hanshu(NUM)=0
    !y(NUM-1)=x(NUM-1)
    !y(NUM)=x(NUM)
    !write(2,*)y(NUM-1),F_hanshu(NUM-1)
    !write(2,*)y(NUM),F_hanshu(NUM)
    
      !======================尾迹
    
    allocate(Fw(Nw),fc(Nc),xc(Nc),xw(Nw))
        do i=1,Nw
              xw(i)=i*0.8*y(Num)/Nw+y(Num)
        end do
        
        
      
        do i=1,Nw
              Fw(i)=0
           do j=1,Num-1
                do kk=1,Nc
              
               xc(kk)=(kk-1)*(y(j+1)-y(j))/(Nc-1)+y(j)
               Fc(kk)=(kk-1)*(F_hanshu(j+1)-F_hanshu(j))/(Nc-1)+F_hanshu(j)
                end do
            
                do kk=1,Nc-1
                     Fw(i)=Fw(i)+0.5*(xc(kk+1)-xc(kk))*(Fc(kk)*(y(Num)-xc(kk))**0.5/(xw(i)-xc(kk))+Fc(kk+1)*(y(Num)-xc(kk+1))**0.5/(xw(i)-xc(kk+1)))
                end do
        end do
    
        Fw(i)=-1*Fw(i)/(pi*(xw(i)-y(Num))**0.5)
        write(2,*)xw(i),fw(i)
        end do
    
    
        do  i=Num+1,Num+Nw
              y(i)=xw(i-Num)
              F_hanshu(i)=fw(i-Num)
        end do
    
!==================================计算F-函数零根
    do i=3,NUM
        if(F(i)*F(i-1)<=0)then
            root=0.5*(y(i)+y(i-1))
            root_i=i
            write(4,*)root
        end if
    end do
!==================================计算delta_P
    allocate(delta_P(NUM+Nw),x1(NUM+Nw))
    write(5,*)" variables=x-Bh,<greek>D</greek>p/p<sub>0</sub> "
    write(5,*) -1,0
    do i=1,NUM+Nw
        delta_P(i)=p0*gamma*M*M/sqrt(2*B*r)*F_hanshu(i)
        delta_P(i)=delta_P(i)/p0
        x1(i)=-k*F_hanshu(i)*sqrt(r)+y(i)
        write(5,*)x1(i),delta_P(i)
    end do
    
!==================================计算P0    
    !open(unit=7,file="root.dat",action="read",status="old")
    !
    !
    !allocate(Sum_F(NUM))
    !Sum_F(1)=0
    !    do i=2,NUM
    !        Sum_F(i)=F_hanshu(i)*(y(i)-y(i-1))+Sum_F(i-1)
    !    end do
    !   
    !
    !
    !open(unit=6,file="DP_x.dat",action="write",status="replace")
    !DP=gamma/sqrt(gamma+1.0)*(2.0*B)**(0.25)*r**(-0.75)*Sum_F(root_i)**0.5
    !XL=B*r+root-r**0.25*(2.0*k*Sum_F(root_i))**0.5
    !XT=B*r+root+r**0.25*(2.0*k*Sum_F(root_i))**0.5
    !write(6,*)XL-(XT-XL)*0.2,0
    !write(6,*)XL,0
    !write(6,*)XL,DP
    !write(6,*)XT,-DP
    !write(6,*)XT,0
    !write(6,*)XT+(XT-XL)*0.2,0
    close(unit=2)
    close(unit=3)
    close(unit=4)
    close(unit=5)
    close(unit=6)
    
    
    end subroutine F_function   
    
    
    
     subroutine F_function2
    implicit none
    real,allocatable::x(:),S(:),F(:),y(:),F_hanshu(:),delta_P(:),x1(:),Sum_F(:)
    real pi,gamma,M,B,r,p0,k,DP,XL,XT,root,L,HL,KG
    real,allocatable::Fw(:),fc(:),xc(:),xw(:)
    integer i,j,NUM,root_i,Nc,Nw,kk
    parameter(pi=3.141592653589793)
    parameter(gamma=1.4)
    open(unit=1,file="sonicboom/2nd_differancial.dat",action="read",status="old")
    open(unit=2,file="sonicboom/F_function.dat",action="write",status="replace")
    open(unit=4,file="sonicboom/root.dat",action="write",status="replace")
    open(unit=5,file="outdata/nearfield.dat",action="write",status="replace")
    open(unit=6,file="indata/FABoom.in",action="read",status="old")
    read(6,*)
    read(6,*)
    read(6,*)
    read(6,*)
    read(6,*)M,p0,p0,p0,p0,HL
    read(6,*)
    read(6,*)L,L
    r=L*HL
    !M=1.26
    !p0=101000
    !r=20
    B=sqrt(M*M-1)
    
    k=1.0/sqrt(2.0)*(gamma+1)*M**4*B**(-3.0/2.0)
    
!==================================读取面积导数行数
    i=0
    do while(.true.)
    read(1,*,end=10) 
         i=i+1
    end do
10  continue  
    
    close(unit=1)
    NUM=i
    allocate(x(NUM),S(NUM))
!==================================读取面积导数    
    open(unit=3,file="sonicboom/2nd_differancial.dat",action="read",status="old")
    rewind(3)
    
    do i=1,NUM
        read(3,*) x(i),S(i)
    end do 
!==================================计算F-函数   
    Nc=10
    Nw=50

    allocate(F(NUM),y(NUM+Nw),F_hanshu(NUM+Nw))
    do j=1,NUM
        y(j)=x(j)
        F(1)=0
        F(2)=S(2)*(x(2)-x(1))/sqrt(x(2)-x(1))
        !F(NUM-1)=0
        !F(NUM)=0
        if(j>2)then
        do i=2,j-1 !2020.8.28修改f函数末端
            !F(i)=F(i-1)+(x(i)-x(i-1))*S(i)/sqrt(y(j)-x(i))
            F(i)=F(i-1)+(x(i)-x(i-1))*(S(i)/sqrt(y(j)-x(i))+S(i-1)/sqrt(y(j)-x(i-1)))*0.5
            
        end do
        !F(j)=2*F(i-1)-F(i-2)
        F(j)=2*F(j-1)-F(j-2)
        end if
        
        F_hanshu(j)=1.0/(2*pi)*F(j)
        write(2,*)y(j),F_hanshu(j)
    end do
    !F_hanshu(NUM-1)=0
    !F_hanshu(NUM)=0
    !y(NUM-1)=x(NUM-1)
    !y(NUM)=x(NUM)
    !write(2,*)y(NUM-1),F_hanshu(NUM-1)
    !write(2,*)y(NUM),F_hanshu(NUM)
    
      !======================尾迹
    
    allocate(Fw(Nw),fc(Nc),xc(Nc),xw(Nw))
        do i=1,Nw
              xw(i)=i*0.8*y(Num)/Nw+y(Num)
        end do
        
        
      
        do i=1,Nw
              Fw(i)=0
           do j=1,Num-1
                do kk=1,Nc
              
               xc(kk)=(kk-1)*(y(j+1)-y(j))/(Nc-1)+y(j)
               Fc(kk)=(kk-1)*(F_hanshu(j+1)-F_hanshu(j))/(Nc-1)+F_hanshu(j)
                end do
            
                do kk=1,Nc-1
                     Fw(i)=Fw(i)+0.5*(xc(kk+1)-xc(kk))*(Fc(kk)*(y(Num)-xc(kk))**0.5/(xw(i)-xc(kk))+Fc(kk+1)*(y(Num)-xc(kk+1))**0.5/(xw(i)-xc(kk+1)))
                end do
        end do
    
        Fw(i)=-1*Fw(i)/(pi*(xw(i)-y(Num))**0.5)
        write(2,*)xw(i),fw(i)
        end do
    
    
        do  i=Num+1,Num+Nw
              y(i)=xw(i-Num)
              F_hanshu(i)=fw(i-Num)
        end do
    
!==================================计算F-函数零根
    do i=3,NUM
        if(F(i)*F(i-1)<=0)then
            root=0.5*(y(i)+y(i-1))
            root_i=i
            write(4,*)root
        end if
    end do
!==================================计算delta_P
    allocate(delta_P(NUM+Nw),x1(NUM+Nw))
    write(5,*)" variables=x-Bh,<greek>D</greek>p/p<sub>0</sub> "
    write(5,*) -1,0
    do i=1,NUM+Nw
        delta_P(i)=p0*gamma*M*M/sqrt(2*B*r)*F_hanshu(i)
        delta_P(i)=delta_P(i)/p0
        x1(i)=y(i)
        write(5,*)x1(i),delta_P(i)
    end do
    
!==================================计算P0    
    !open(unit=7,file="root.dat",action="read",status="old")
    !
    !
    !allocate(Sum_F(NUM))
    !Sum_F(1)=0
    !    do i=2,NUM
    !        Sum_F(i)=F_hanshu(i)*(y(i)-y(i-1))+Sum_F(i-1)
    !    end do
    !   
    !
    !
    !open(unit=6,file="DP_x.dat",action="write",status="replace")
    !DP=gamma/sqrt(gamma+1.0)*(2.0*B)**(0.25)*r**(-0.75)*Sum_F(root_i)**0.5
    !XL=B*r+root-r**0.25*(2.0*k*Sum_F(root_i))**0.5
    !XT=B*r+root+r**0.25*(2.0*k*Sum_F(root_i))**0.5
    !write(6,*)XL-(XT-XL)*0.2,0
    !write(6,*)XL,0
    !write(6,*)XL,DP
    !write(6,*)XT,-DP
    !write(6,*)XT,0
    !write(6,*)XT+(XT-XL)*0.2,0
    close(unit=2)
    close(unit=3)
    close(unit=4)
    close(unit=5)
    close(unit=6)
    
    
    end subroutine F_function2
   
    
    !========================================================================================================！
    !                                                                                                        ！
    !                                       激波修正程序shock_correct                                        ！
    !                                       激波修正程序shock_correct                                        ！
    !                                       激波修正程序shock_correct                                        ！    
    !                                        2020.7.8更新完整子程序                                          ！
    !                          2020.8.6添加面积修正判断条件的小激波无法处理的情况                            ！
    !========================================================================================================！       
    subroutine shock_correct
    implicit none
    !real,allocatable::x1(:),delta_p1(:),x2(:),delta_p2(:),x(:),delta_p(:),S1(:),S2(:),rank(:),xc(:),delta_pc(:)
    real,allocatable::x0(:),dp0(:),x1(:),dp1(:),dtx(:),xc(:),dpc(:),rank(:),xcr(:),dpcr(:),S1(:),S2(:),x(:),dp(:)
    integer,allocatable::crosspoint(:,:)
    integer i,j,k,n0,n1,nc,ncr,cross,n,check,seed
    real dx,shock,paixu,seedmod,seedx,seeddp,temp
    open(unit=1,File="sonicboom/delta_P.dat",action="read",status="old")
    open(unit=2,File="sonicboom/delta_PC.dat",action="write",status="replace")
    open(unit=3,File="outdata/nearfield signal.dat",action="write",status="replace")
    !open(unit=4,File="temp.dat",action="write",status="replace")
    !========================================== 读取原文件点数
    i=0
    do while(.TRUE.)
        read(1,*,end=1)
        i=i+1
    end do
1   continue
    rewind(1)
    n0=i-1         !原文件点数
    !write(*,*)n0
    read(1,*)
    !========================================== 读取原文件数据
    allocate(x0(n0),dp0(n0),rank(n0))        !排序数组与x0数组相同
    do i=1,n0
        read(1,*)x0(i),dp0(i)
        !write(3,*)x0(i),dp0(i)
        rank(i)=x0(i)
    end do
    !write(*,*)rank(1),rank(n0)
    !========================================== 找原数据头尾值，最小值和最大值
    do i=1,n0-1
        do j=i+1,n0
            if(rank(i)>rank(j))then
                paixu=rank(i)
                rank(i)=rank(j) 
                rank(j)=paixu
            end if
        end do
    end do
    !write(*,*)rank(n0),rank(1)

  !==========================================延拓X区间，将值赋值到x1和dp1数组
    n1=n0+2
    allocate(x1(n1),dp1(n1))
    x1(1)=rank(1)-1
    x1(n1)=rank(n0)+1
    dp1(1)=dp0(1)
    dp1(n1)=dp0(n0)
    !write(2,*)n1!x1(1),dp1(1)
    !write(*,*)n1
    do i=2,n1-1
        x1(i)=x0(i-1)
        dp1(i)=dp0(i-1)
        !write(2,*)x1(i),dp1(i)
    end do
   ! write(*,*)"波形延拓完成"
  !==========================================求每两个点之间的长度并确定插值步长
  !allocate(dtx(n1-1))
  !do i=1,n1-1
  !    dtx(i)=abs(x1(i+1)-x1(i))
  !end do
  !
  !do i=1,n1-2
  !    do j=i+1,n1-1
  !        if(dtx(i)>dtx(j))then
  !            paixu=dtx(i)
  !            dtx(i)=dtx(j)
  !            dtx(j)=paixu
  !        end if
  !    end do
  !end do
  !dx=dtx(1)
 ! dx=0.1
  !write(*,*)"OK"!dx
          
  !========================================== 插值步长
    !write(*,*)"输入步长按1,默认按2"
    !read(*,*)i
    !if(i==1)then
    !    write(*,*)"输入步长"
    !    read(*,*)dx
    !else
    !    dx=abs(x1(n1)-x1(1))/100
    !end if
    !if(dx<=1e-2)then
    !    dx=1e-2
    !end if
    !!write(*,*)dx
   !========================================== 插值用xc去插值x1
   !dx=0.1
   nc=1000!int((x1(n1)-x1(1))/dx)
  ! write(*,*)nc
   allocate(xc(nc),dpc(nc))
   xc(1)=x1(1)
   do i=1,nc
       xc(i)=xc(1)+(x1(n1)-x1(1))/(nc-1)*(i-1)
       !write(*,*)xc(i)
   end do
   
   do i=1,n1-1
       if(x1(i)<=x1(i+1))then
           do j=1,nc
               if(xc(j)>=x1(i).AND.xc(j)<x1(i+1))then
                   dpc(j)=dp1(i)+(dp1(i+1)-dp1(i))/(x1(i+1)-x1(i))*(xc(j)-x1(i))
                   write(2,*)xc(j),dpc(j)
               end if
           end do
       else
           do j=nc,1,-1
               if(xc(j)<=x1(i).AND.xc(j)>x1(i+1))then
                   dpc(j)=dp1(i)+(dp1(i+1)-dp1(i))/(x1(i+1)-x1(i))*(xc(j)-x1(i))
                   write(2,*)xc(j),dpc(j)
               end if
           end do
       end if
   end do
   !write(*,*)"波形插值完成"
   !========================================== 插值用xc去插值x1
   !!temp=mod(-3.6,1.0)
   !!write(*,*)temp
   !do i=1,n1-1
   !    seed=int(abs(x1(i+1)-x1(i))/dx)
   !    seedmod=mod((x1(i+1)-x1(i)),dx)
   !    !write(*,*)i!seed,seedmod
   !    write(2,*)x1(i),dp1(i)
   !    if(seedmod/=0.AND.seed/=0)then   !非整数个节点区间
   !        do j=1,seed
   !            !seedx=x1(i)+dx*j
   !            seedx=x1(i)+(x1(i+1)-x1(i)-seedmod)/seed*j
   !            !seeddp=dp1(i)+dx*j*(dp1(i+1)-dp1(i))/(x1(i+1)-x1(i))
   !            seeddp=dp1(i)+(dp1(i+1)-dp1(i))/(x1(i+1)-x1(i))*(x1(i+1)-x1(i)-seedmod)/seed*j
   !            write(2,*)seedx,seeddp
   !        end do
   !    else if(seedmod==0.AND.seed/=1)then !整数个节点区间，排除1个区间
   !        do j=1,seed-1
   !            seedx=x1(i)+(x1(i+1)-x1(i)-seedmod)/seed*j
   !            seeddp=dp1(i)+(dp1(i+1)-dp1(i))/(x1(i+1)-x1(i))*(x1(i+1)-x1(i)-seedmod)/seed*j
   !            write(2,*)seedx,seeddp
   !        end do
   !    end if
   !end do
   !write(2,*)x1(n1),dp1(n1)
   !write(*,*)"波形插值完成"
       
   !============================================读取插值文件和数据
   close(2)
   open(unit=4,File="sonicboom/delta_PC.dat",action="read",status="old")
   i=0
    do while(.true.)
        read(4,*,end=2)
        i=i+1
    end do
2  continue
    rewind(4)
    ncr=i
    !write(*,*)ncr
    allocate(xcr(ncr),dpcr(ncr))
    do i=1,ncr
        read(4,*)xcr(i),dpcr(i)      !x chazhi  read
    end do
    !write(*,*)"插值波形读取完成"
   !============================================建立X方向搜索数组
   n=2*nc
   allocate(x(n))
   do i=1,n
       x(i)=xcr(1)+(xcr(ncr)-xcr(1))/(n-1)*(i-1)
       !write(*,*)x(i)
   end do
   !write(*,*)x(1),x(n)
   !=============================================面积平衡算法开始
   !=============================================面积平衡算法开始
   !=============================================面积平衡算法开始
   
   !=============================================查找前三个交点
   !write(*,*)"开始面积平衡修正"
    allocate(crosspoint(n,3),S1(n),S2(n))
    S1(:)=0
    S2(:)=0
    do while(.True.)
        check=0   !有无三点相交
        
        do i=1,n
            cross=0
            do j=1,ncr-1
                if(((xcr(j)<=x(i).AND.x(i)<xcr(j+1)).OR.(xcr(j)>=x(i).AND.x(i)>xcr(j+1))).AND.(xcr(j)/=xcr(j+1)))then
                !if((xcr(j)-x(i))*(xcr(j+1)-x(i))<=0)then
                    !write(*,*)"老子进来了"
                    cross=cross+1           !》》》》》》》当前位置交点加一
                    crosspoint(i,cross)=j     !》》》》》》》记录交点位置j计数器
                end if
                if(cross==3)then          !》》》》》》发现3个交点就停止搜索跳出j循环，开始算当前的面积    
                    check=check+1
                    !write(*,*)x(i)
                    go to 4
                end if
            end do               !结束j的遍历
4           continue
            !开始求解两个面积，这个时候搜索i还没有动
            if(cross==3)then
            
            do j=crosspoint(i,1)+1,crosspoint(i,2)
                S1(i)=(xcr(j)+xcr(j-1)-2*xcr(crosspoint(i,2)))*(dpcr(j)-dpcr(j-1))*0.5+S1(i)
                !S1(i)=(xcr(j)+xcr(j-1)-2*x(i))*(dpcr(j)-dpcr(j-1))*0.5+S1(i)
            end do
            do j=crosspoint(i,2)+1,crosspoint(i,3)
                S2(i)=(2*xcr(crosspoint(i,2))-xcr(j)-xcr(j-1))*(dpcr(j)-dpcr(j-1))*0.5+S2(i)
            end do
            
            !write(*,*)x(i),S1(i),S2(i)
            !
            !
            
            !if(((S1(i)-S2(i))<0.AND.(S1(i-1)-S2(i-1))>0).OR.(S1(i)==S1(i-1).AND.S2(i)==S2(i-1).AND.crosspoint(i,3)==(crosspoint(i,1)+2)))then  !2020.8.5加入面积修正的条件
            !if(((S1(i)-S2(i))<0.AND.(S1(i-1)-S2(i-1))>0).OR.((crosspoint(i,2)==(crosspoint(i,1)+2)).AND.(crosspoint(i,3)==(crosspoint(i,2)+2))))then  !2020.8.5加入面积修正的条件
            if(((S1(i)-S2(i))<0.AND.(S1(i-1)-S2(i-1))>0).OR.   &
     &         ((S1(i)<S2(i)).AND.(S1(i-1)==0).AND.(S2(i-1)==0)))then   !2020.8.6日修改，由于一些特殊波形会出现S2直接比S1大，故添加此条件
                !write(*,*)"OK",x(i)
                xcr(crosspoint(i-1,1):crosspoint(i,3)+1)=(x(i)+x(i-1))*0.5 
                dpcr(crosspoint(i-1,1):crosspoint(i,3)+1)=dpcr(crosspoint(i-1,1))!(dpcr(crosspoint(i-1,1))+dpcr(crosspoint(i,3)+1))*0.5!Y值全往下缩
                !S1(:)=0
                !S2(:)=0

                goto 5
            end if
            end if
          
            
            
            
            
            
            
        end do
        
5       continue      
        !write(*,*)check
    !
    ! write(3,*)"variables=x-Bh,<greek>D</greek>p/p<sub>0</sub> "
    !do i=1,ncr
    !    write(3,*) xcr(i),dpcr(i)
    !end do   
        !write(*,*)check 
       write(*,*)"还剩下",check,"个多值点"
        if(check==0)then
            go to 3
        end if
         

    end do
 3  continue   
     write(3,*)"variables=x-Bh,<greek>D</greek>p/p<sub>0</sub> "
     write(3,*) xcr(1)-10,dpcr(1)
    do i=2,ncr
        if(xcr(i)/=xcr(i-1))then
        write(3,*) xcr(i),dpcr(i)
        end if
    end do   
    
    close(unit=1)
    close(unit=2)
    close(unit=3)
    close(unit=4)
    
  
    end subroutine shock_correct
    
    
    
    
    
    !========================================================================================================!
    !                                                                                                        !
    !   FUNC.:     determine the shock position using Bugers-Hayes method                                    !  
    !   AUTHOR:    qiao jianling                                                                             !
    !   DATE:      2021.11.24                                                                                !
    !                                                                                                        !
    !========================================================================================================!       
    subroutine shock_correct_BH
    implicit double precision(a-h,o-z)
    allocatable x_read(:),dp_read(:),interp_points(:)
    allocatable x_d(:),dp_d(:),phi(:)
    allocatable x(:),dp(:)
    parameter ( eps = 1.e-6 )
    parameter ( np  = 10001 )
    
    allocate( x(np),dp(np) )
    
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>read the distorted signature from the file = sonicboom/delta_P.dat>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    open(521,file='sonicboom/delta_P.dat')
    read(521,*)
    icount = 0
    do while( .true. )
        read(521,*,end=100)
        icount = icount + 1
    enddo
100 continue
    rewind(521)
    
    num_read = icount + 2   ! extern the signature
    allocate( x_read(num_read),dp_read(num_read),interp_points(num_read) )
    read(521,*)
    do i = 2,num_read-1
        read(521,*)x_read(i),dp_read(i)
    enddo
    close(521)
    x_read(1)         = minval(x_read(2:num_read-1))-eps
    x_read(num_read)  = maxval(x_read(2:num_read-1))+eps
    x_read(num_read)  = x_read(num_read)+0.1*(x_read(num_read)-x_read(1))
    dp_read(1)        = 0.0
    dp_read(num_read) = 0.0
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<read the distorted signature from the file = sonicboom/delta_P.dat<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>generate the x coordinate of the modified signature>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    dx = ( x_read(num_read)-x_read(1) )/(np-1)
    x(1) = x_read(1)
    do i = 2,np
        x(i) = x(i-1)+dx
    enddo
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<generate the x coordinate of the modified signature<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>interp the distorted signature for calcualting phi>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    interp_points = 1
    do i = 2,num_read
        ds = sqrt((x_read(i)-x_read(i-1))**2+(dp_read(i)-dp_read(i-1))**2)
        if( ds>dx ) interp_points(i) = int(ds/dx)
    enddo
    num = sum(interp_points)
    allocate( x_d(num),dp_d(num) )
    
    j = 1
    x_d(1) = x_read(1)
    dp_d(1)= dp_read(1)
    do i = 2,num_read
        if( interp_points(i)>1 )then
            ipoints = interp_points(i)
            dx      = (x_read(i)-x_read(i-1))/ipoints
            gre_dp  = (dp_read(i)-dp_read(i-1))/(x_read(i)-x_read(i-1))
            do k = 1,ipoints-1
                j = j+1
                x_d(j)  = x_d(j-1)+dx
                dp_d(j) = dp_d(j-1)+dx*gre_dp
            enddo
        endif
        j = j+1
        x_d(j)  = x_read(i)
        dp_d(j) = dp_read(i)
    enddo
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<interp the distorted signature for calcualting phi<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!

    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>calculate the function phi of the distorted signautre>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
    allocate( phi(num) )
    phi = 0.0
    do i = 2,num
        phi(i) = phi(i-1) + 0.5*( dp_d(i) + dp_d(i-1) )*( x_d(i) - x_d(i-1) )
    enddo
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<calculate the function phi of the distorted signautre<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!
    
    call distortion_to_uniform(x_d,dp_d,phi,num,x,dp,np)
    
    open(520,file='outdata/nearfield.dat')
    write(520,*)'variables=x-Bh,<greek>D</greek>p/p<sub>0</sub>'
    do i = 1,np
        write(520,'(2G18.10)')x(i),dp(i)
    enddo
    close(520)
    
    return
    end subroutine shock_correct_BH
    
    !========================================================================================================!
    !                                                                                                        !
    !   FUNC.:     determine the shock position using Bugers-Hayes method                                    !  
    !              将在扭曲网格点上的波形，插值回均匀网格点上，当出现多值情况时，采用BH方法确定激波          !
    !              from the bBoom project                                                                    !
    !   AUTHOR:    qiao jianling                                                                             !
    !   DATE:      2021.11.24                                                                                !
    !                                                                                                        !
    !========================================================================================================!
    subroutine distortion_to_uniform(x_d,dp_d,phi,num,x,dp,np)
    implicit double precision (a-h,o-z)
    dimension x_d(num),dp_d(num),phi(num)
    dimension x(np),dp(np)
    allocatable dp_fore(:),dp_back(:)
    allocatable phi_fore(:),phi_back(:)
    allocatable j_xd_fore(:),j_xd_back(:),theta(:)
    logical iflag
    
    allocate( dp_fore (np),dp_back (np) )
    allocate( phi_fore(np),phi_back(np) )
    allocate( j_xd_fore(np),j_xd_back(np) )
    allocate( theta(num) )
    
    theta(2:num) = x_d(2:num)-x_d(1:num-1)
    theta(1)    = 0.0
    
    call interp_fore(x_d,dp_d,phi,num,x,dp_fore,phi_fore,j_xd_fore,np)
    
    call interp_back(x_d,dp_d,phi,num,x,dp_back,phi_back,j_xd_back,np)
    
    dp = 0.0
    iflag = .false.
    
    dp(1)  = 0.5*(dp_fore(1)+dp_back(1))
    dp(np) = 0.5*(dp_fore(np)+dp_back(np))
    
    do i = 2,np-1
        if( .not. iflag ) i_temp1 = i-1
        i_temp2 = i
        delta_phi = abs(phi_back(i)-phi_fore(i))
        if( delta_phi>1.d-16 )then
            !*****there must be multivalue*****!
            iflag = .true.
        else
            !*****there may be multivalue******!
            iflag = .false.
        endif
        
        if( .not. iflag )then
            jf = j_xd_fore(i_temp1)
            jb = j_xd_back(i_temp2)
            if( minval(theta(jf+1:jb))>=0.0 )then
                !****there is not multivalue within [I_TEMP1,I_TEMP2]****!
                dp(i_temp2) = 0.5*(dp_fore(i_temp2)+dp_back(i_temp2))
            else
                !****there must be multivalue within [I_TEMP1,I_TEMP2]****!
                k1 = jf
                k2 = jf+1
                do j = i_temp1+1,i_temp2-1
                    k1_old  = k1
                    k2_old  = k2
                    phi_max = -1.0d16
                    dp_tmp  = dp_fore(j)
                    !****step1: interpolate PHI on the point J, maximize PHI, record the location on the distrod grid****!
                    do k = jf,jb
                        if( x(j)>=x_d(k) .and. x(j)<=x_d(k+1) )then
                            gre_phi = (phi(k+1)-phi(k))/(x_d(k+1)-x_d(k))
                            phi_tmp = phi(k)+(x(j)-x_d(k))*gre_phi
                            gre_dpd = (dp_d(k+1)-dp_d(k))/(x_d(k+1)-x_d(k))
                            dpd_tmp = dp_d(k)+(x(j)-x_d(k))*gre_dpd
                            if( phi_tmp>phi_max )then
                                phi_max = phi_tmp
                                k1      = k
                                k2      = k+1
                                dp_tmp  = dpd_tmp
                            endif
                        endif
                    enddo
                    
                    !****step2: judge the multivalue phenomenon within [K1_OLD,K2]****!
                    if( minval(theta(k1_old+1:k2))>=0.0 )then
                        !****there is not multivalue****!
                        dp(j) = dp_tmp
                    else
                        !****there must be multivalue, but not treatment because the value of dx cannot discrib the shock position****!
                        dp(j) = dp_tmp
                    endif
                enddo
                
                !****step3: judge the multivalue phenomenon within [K1,JB]****!
                if( minval(theta(k1+1:jb))>=0.0 )then
                    !****there is not multivalue****!
                    dp(i_temp2) = 0.5*(dp_fore(i_temp2)+dp_back(i_temp2))
                else
                    !****there must be multivalue, but not treatment because the value of dx cannot discrib the shock position****!
                    dp(i_temp2) = 0.5*(dp_fore(i_temp2)+dp_back(i_temp2))
                endif
                
            endif
        endif
    enddo
    
    end subroutine distortion_to_uniform
    
!=================================================================================================
!
!     2020.11.24 沿tau增大的方向插值，即从左向右插值 from the bBoom project
!    
!     input：
!            TAUD,PK,PHI,TAU,NP
!     output:
!            PKK_FORE,PHI_FORE,J_TAUD_FORE
!
!================================================================================================= 
    SUBROUTINE INTERP_FORE(TAUD,PK,PHI,NUM,TAU,PKK_FORE,PHI_FORE,J_TAUD_FORE,NP)
    IMPLICIT DOUBLE PRECISION(A-H,O-Z)
    DIMENSION TAUD(NUM),PK(NUM),PHI(NUM)
    DIMENSION TAU(NP),PKK_FORE(NP),PHI_FORE(NP),J_TAUD_FORE(NP)
    
    J = 1
    DO I = 1,NP
        IF( TAU(I)<=TAUD(1) )THEN
            PHI_FORE(I) = PHI(1)
            PKK_FORE(I) = PK(1)
            J_TAUD_FORE(I) = 1
        ELSEIF( TAU(I)>=TAUD(NUM) )THEN
            PHI_FORE(I) = PHI(NUM)
            PKK_FORE(I) = PK(NUM)
            J_TAUD_FORE(I) = NUM
        ELSE
            DO WHILE( J<NUM )
                IF( TAU(I)>=TAUD(J) .AND. TAU(I)<TAUD(J+1) )THEN
                    GRE_PHI     = (PHI(J+1)-PHI(J))/(TAUD(J+1)-TAUD(J))
                    PHI_FORE(I) = PHI(J)+(TAU(I)-TAUD(J))*GRE_PHI
                    GRE_DP      = (PK(J+1)-PK(J))/(TAUD(J+1)-TAUD(J))
                    PKK_FORE(I) = PK(J)+(TAU(I)-TAUD(J))*GRE_DP
                    
                    J_TAUD_FORE(I) = J
                    EXIT
                ELSE
                    J      = J+1
                ENDIF
            ENDDO
        ENDIF
    ENDDO
    
    RETURN
    END SUBROUTINE INTERP_FORE
    
!=================================================================================================
!
!     2020.11.24 沿tau减小的方向插值，即从右向左插值 from the bBoom project
!    
!     input：
!            TAUD,PK,PHI,TAU,NP
!     output:
!            PKK_BACK,PHI_BACK,J_TAUD_BACK
!
!================================================================================================= 
    SUBROUTINE INTERP_BACK(TAUD,PK,PHI,NUM,TAU,PKK_BACK,PHI_BACK,J_TAUD_BACK,NP)
    IMPLICIT DOUBLE PRECISION(A-H,O-Z)
    DIMENSION TAUD(NUM),PK(NUM),PHI(NUM)
    DIMENSION TAU(NP),PKK_BACK(NP),PHI_BACK(NP),J_TAUD_BACK(NP)
    
    J = NUM
    DO I = NP,1,-1
        IF( TAU(I)<=TAUD(1) )THEN
            PHI_BACK(I) = PHI(1)
            PKK_BACK(I) = PK(1)
            J_TAUD_BACK(I) = 1
        ELSEIF( TAU(I)>=TAUD(NUM) )THEN
            PHI_BACK(I) = PHI(NUM)
            PKK_BACK(I) = PK(NUM)
            J_TAUD_BACK(I) = NUM
        ELSE
            DO WHILE( J>1 )
                IF( TAU(I)>TAUD(J-1) .AND. TAU(I)<=TAUD(J) )THEN
                    GRE_PHI     = (PHI(J)-PHI(J-1))/(TAUD(J)-TAUD(J-1))
                    PHI_BACK(I) = PHI(J-1)+(TAU(I)-TAUD(J-1))*GRE_PHI
                    GRE_DP      = (PK(J)-PK(J-1))/(TAUD(J)-TAUD(J-1))
                    PKK_BACK(I) = PK(J-1)+(TAU(I)-TAUD(J-1))*GRE_DP
                    
                    J_TAUD_BACK(I) = J
                    EXIT
                ELSE
                    J      = J-1
                ENDIF
            ENDDO
        ENDIF
    ENDDO
    
    RETURN
    END SUBROUTINE INTERP_BACK
    
    
    !========================================================================================================!
    !                                                                                                        !
    !   FUNC.:     Spline Interpolation function for determine the point of intersection                     ! 
    !   AUTHOR:    qiao jianling                                                                             !
    !   DATE:      2021.12.13                                                                                !
    !                                                                                                        !
    !========================================================================================================!
    subroutine spline(x,y,num,xx,yy,k)
    implicit double precision(a-h,o-z)
    dimension x(num),y(num)
    dimension h(2:num),hy(2:num),f(num),dM(num),dL(2:num),dD(num),dU(num-1)
    parameter( bound1 = 0, boundn = 0 )
    
    h  = x(2:num)-x(1:num-1)
    hy = y(2:num)-y(1:num-1)
    
    dL(2:num-1) = h(3:num)/(h(2:num-1)+h(3:num))
    dL(num)     = 1.d0
    
    dU(1)       = 1.d0
    dU(2:num-1) = 1.d0-dL(2:num-1)
    
    dD          = 2.d0
    
    f(2:num-1)  = 3.d0*( dU(2:num-1)*hy(3:num)/h(3:num) + dL(2:num-1)*hy(2:num-1)/h(2:num-1) )
    f(1)        = 3.d0*hy(2)/h(2)-0.5*h(2)*bound1
    f(num)      = 3.d0*hy(num)/h(num)+0.5*h(num)*boundn
    
    call Three_Diag_Equ(dL,dD,dU,f,num,dM)
    
    if( k>=0 )then
    ! calculate the points
    i = k+1
    !do i = 2,num
        !if( xx>=x(i-1) .and. xx<=x(i) )then
            yy = (xx-x(i))**2*(h(i)+2.*(xx-x(i-1)))/(h(i)**3)*y(i-1)
            yy = yy+(xx-x(i-1))**2*(h(i)+2*(x(i)-xx))/(h(i)**3)*y(i)
            yy = yy+(xx-x(i))**2*(xx-x(i-1))/(h(i)**2)*dM(i-1)
            yy = yy+(xx-x(i-1))**2*(xx-x(i))/(h(i)**2)*dM(i)
        !endif
    !enddo
    else
    ! calculate the area
    S = 0.d0
    do i = 2,num
        c1 = y(i-1)/h(i)**3
        c2 = y(i)/h(i)**3
        c3 = dM(i-1)/h(i)**2
        c4 = dM(i)/h(i)**2
        
        A1 = 0.5*x(i)**4 + (h(i)-2.*x(i-1)-4.*x(i))/3.*x(i)**3 + &
             x(i)*( 2.*x(i-1)-h(i)+x(i) )*x(i)**2 + x(i)**2*(h(i)-2.*x(i-1))*x(i)
        A2 = 0.5*x(i-1)**4 + (h(i)-2.*x(i-1)-4.*x(i))/3.*x(i-1)**3 + &
             x(i)*( 2.*x(i-1)-h(i)+x(i) )*x(i-1)**2 + x(i)**2*(h(i)-2.*x(i-1))*x(i-1)
        S  = S+c1*(A1-A2)
        
        A1 = -0.5*x(i)**4 + (h(i)+2.*x(i)+4.*x(i-1))/3.*x(i)**3 &
             -x(i-1)*( 2.*x(i)+h(i)+x(i-1) )*x(i)**2 + x(i-1)**2*(h(i)+2.*x(i))*x(i)
        A2 = -0.5*x(i-1)**4 + (h(i)+2.*x(i)+4.*x(i-1))/3.*x(i-1)**3 &
             -x(i-1)*( 2.*x(i)+h(i)+x(i-1) )*x(i-1)**2 + x(i-1)**2*(h(i)+2.*x(i))*x(i-1)
        S  = S+c2*(A1-A2)
        
        A1 = 0.25*x(i)**4 - (x(i-1)+2*x(i))/3.*x(i)**3 + 0.5*(2.*x(i)*x(i-1)+x(i)**2)*x(i)**2 - x(i)**2*x(i-1)*x(i)
        A2 = 0.25*x(i-1)**4 - (x(i-1)+2*x(i))/3.*x(i-1)**3 + 0.5*(2.*x(i)*x(i-1)+x(i)**2)*x(i-1)**2 - x(i)**2*x(i-1)*x(i-1)
        S  = S+c3*(A1-A2)
        
        A1 = 0.25*x(i)**4 - (x(i)+2*x(i-1))/3.*x(i)**3 + 0.5*(2.*x(i)*x(i-1)+x(i-1)**2)*x(i)**2 - x(i-1)**2*x(i)*x(i)
        A2 = 0.25*x(i-1)**4 - (x(i)+2*x(i-1))/3.*x(i-1)**3 + 0.5*(2.*x(i)*x(i-1)+x(i-1)**2)*x(i-1)**2 - x(i-1)**2*x(i)*x(i-1)
        S  = S+c4*(A1-A2)
    enddo
    yy = ABS(S)+yy
    endif
    
    end subroutine spline
    
    
!本程序是用来解三对角方程组的子例行程序
    subroutine Three_Diag_Equ(L,M,U,Y,n,x)
!L-三对角矩阵的下对角
!M-三对角矩阵的对角线
!U-三对角矩阵的上对角
!Y-方程组的常数项
!n-方程组的维度
!x-方程组的解
    implicit none
    double precision L(n-1),M(n),U(n-1),Y(n)
    double precision a(2:n),b(n),c(n-1),f(n),x(n)
    integer n,i
    a=L;b=M;c=U;f=Y
    b(1)=b(1)
    c(1)=c(1)/b(1)
    f(1)=f(1)/b(1)
    do i=2,n
        a(i)=a(i)
        b(i)=b(i)-a(i)*c(i-1)
        if(i<n)then
            c(i)=c(i)/b(i)
        endif
        f(i)=(f(i)-a(i)*f(i-1))/b(i)
    enddo
    x(n)=f(n)
    do i=n-1,1,-1
        x(i)=f(i)-c(i)*x(i+1)
    enddo
    end subroutine
    