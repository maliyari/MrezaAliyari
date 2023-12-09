clearvars 
clc
close all
% Solving The Laplace Equation For  Obtaining Streamline Field in a 2D Forward Step
% Specifying parameters
X1 = 95.0 ;
X2 = 57.0 ;
H1 = 45 ;
H2 = 27 ;
Q = 45.0 ;
U_INLET = Q/(H1*1.0) ;
U_OUTLET = Q/(H2*1.0) ;
 
NX = 153;
NY = 46 ;
 
 
DEL_Y = H1/(NY-1) ;
DEL_X = (X1+X2)/(NX-1) ;
BETA = (DEL_X/DEL_Y).^2 ;
clearvars 
clc
close all
% Solving The Laplace Equation For  Obtaining Streamline Field in a 2D Forward Step
% Specifying parameters
X1 = 95.0 ;
X2 = 57.0 ;
H1 = 45 ;
H2 = 27 ;
Q = 45.0 ;
U_INLET = Q/(H1*1.0) ;
U_OUTLET = Q/(H2*1.0) ;
 
NX = 153;
NY = 46 ;
 
 
DEL_Y = H1/(NY-1) ;
DEL_X = (X1+X2)/(NX-1) ;
BETA = (DEL_X/DEL_Y).^2 ;

% Applying Initial Value to Variables
ERROR = 1.0 ;
 
INITIAL_S = zeros(NY,NX) ;
FINAL_S = zeros(NY,NX) ;
EKHTELAF_S = zeros(NY,NX) ;
U = zeros(NY,NX) ;
V = zeros(NY,NX) ;

% Grid
Y = 0 : DEL_Y : H1 ;
Y = Y' ;
Y = repmat (Y ,1 , NX);
 
X = 0 : DEL_X : (X1+X2) ;
X = repmat(X,NY,1) ;

% Inlet Boundary Condition
for I = 2 : NY 
    INITIAL_S(I , 1) = INITIAL_S(I-1,1)+(U_INLET*DEL_Y) ;
    FINAL_S(I,1) = FINAL_S(I-1,1)+(U_INLET*DEL_Y) ;
end
% Top Boundary Condition
INITIAL_S(NY , :) = INITIAL_S(NY , 1) ;
FINAL_S(NY , :) = FINAL_S(NY , 1) ;
 
 
% Outlet Boundary Condition
for I = NY : -1 : ((NY -(H2/DEL_Y))+1)
    
    INITIAL_S(I-1 , NX) = INITIAL_S(I , NX)-(U_OUTLET*DEL_Y) ;
    FINAL_S(I-1 , NX) = FINAL_S(I , NX)-(U_OUTLET*DEL_Y) ;
end

% Iterative Solving the Streamline Field
while ERROR > 1e-10
    
    INITIAL_S = FINAL_S ;
    
    for I = 2 : NY-1 
        for J = 2 : NX-1
            
            if X(I,J) < X1 
                FINAL_S(I,J) = (INITIAL_S(I,J+1)+INITIAL_S(I,J-1)+...
                    (BETA*(INITIAL_S(I-1,J)+INITIAL_S(I+1,J))))/(2*(1+BETA)) ;
                
            else if (X(I,J) >= X1 && Y(I,J) > (H1-H2))
                    
                    FINAL_S(I,J) = (INITIAL_S(I,J+1)+INITIAL_S(I,J-1)+...
                        (BETA*(INITIAL_S(I-1,J)+INITIAL_S(I+1,J))))/(2*(1+BETA)) ;
                end
            end
        end
end
 
 
EKHTELAF_S = FINAL_S - INITIAL_S ;
 
ERROR = max(max(abs(EKHTELAF_S))) ;
 
end
% Calculating the Velocity Field
 U(2:NY,1) = U_INLET ;
 
for I = 2 : NY
   for J = 2 : NX
       
    U(I,J) = (FINAL_S(I,J)-FINAL_S(I-1,J))/DEL_Y ;
    V(I,J) = -(FINAL_S(I,J)-FINAL_S(I,J-1))/DEL_X ; 
    
   end
end
% Exporting Output for Tecplot
for i = 1 : (NX)*(NY)
    
            a(i,1)=X(i);
            a(i,2)=Y(i);
            a(i,3)=U(i);
            a(i,4)=V(i);
            
end
% Find indices where U is zero
zero_indices = U == 0;

% Set the corresponding values in X, Y, and U to NaN
X(zero_indices) = NaN;
Y(zero_indices) = NaN;
U(zero_indices) = NaN;

% Assuming X, Y, and U are already defined as 46x153 matrices

figure(1)
% Create a 3D surface plot
surf(X, Y, U, 'EdgeColor', 'none');

% Set colormap to jet
colormap(turbo(256));

% Add labels and title
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Velocity (U)');
title('Vertical Velocity');

xticks(0:4:152)

% Add colorbar for reference
clim([0 3])
colorbar;
axis equal tight;

% Adjust the view for better visualization (optional)
view(2); % 2D view from the top

grid off;

figure(2)
[startX,startY] = meshgrid(0,0:1:45);
h = streamline(0:1:152, 0:1:45, U, V, startX, startY);
set(h, 'Color', 'black', 'LineWidth', 1.5); % Customize streamline appearance
axis equal tight;
xticks(0:4:152)
title('Streamlines')
xlabel('X (m)')
ylabel('Y (m)')




