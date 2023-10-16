clear all; clc;

t0 = 0; tf = 10; dt = 0.1;
tspan = t0:dt:tf;
t_info = [t0, tf, dt];
y0 = [1; 0];

adamc = table2array(readtable('adams1.xlsx'));
adamc2 = table2array(readtable('adams2.xlsx'));

y_true = [cos(tspan); sin(tspan)];
y_rk78 = inte_rk8(tspan, y0, @dynamic);

[t, y_rk4] = RK4(@dynamic, tspan, y0, dt); 

n = 6;
y_adams = adams(tspan, y0, @dynamic, n, adamc);
y_pece = pece(tspan, y0, @dynamic, n, adamc, adamc2);
y_pemcme = pemcme(tspan, y0, @dynamic, n, adamc, adamc2);
y_pe2ce = pe2ce(tspan, y0, @dynamic, n, adamc, adamc2);


figure(1)
hold on
plot(tspan, log10(abs(y_true(2,:)-y_rk78(2,:))));
plot(tspan, log10(abs(y_true(2,:)-y_rk4(2,:))));
plot(tspan, log10(abs(y_true(2,:)-y_adams(2,:))));
plot(tspan, log10(abs(y_true(2,:)-y_pece(2,:))));
plot(tspan, log10(abs(y_true(2,:)-y_pemcme(2,:))));
plot(tspan, log10(abs(y_true(2,:)-y_pe2ce(2,:))));

legend('RK78', 'RK4', 'adams', 'pece', 'pemcme', 'pe2ce', Location='best')
box on; grid on;
set(0,'defaultLineLineWidth',1.5);
set(0,'defaultfigurecolor','w');
set(gca,'linewidth',1.2,'fontsize',15);

function [t,y] = RK4(fun,t,y0,h)
    m=length(y0);
    n=length(t);
    y=zeros(m,n);
    y(:,1) = y0;
    
    for i=1:n-1
        t(i+1)=t(i)+h;
        k1=fun(t(i),y(:,i));
        k2=fun(t(i)+h/2,y(:,i)+h*k1/2);
        k3=fun(t(i)+h/2,y(:,i)+h*k2/2);
        k4=fun(t(i)+h,y(:,i)+h*k3);
        y(:,i+1)=y(:,i)+h*(k1+2*k2+2*k3+k4)/6;
    end
end
function dy = dynamic(t, y)
    dy = [-y(2); y(1)];
end

function y = adams(tspan, x0, f, n, adamc)
    tn = length(tspan);
    y = zeros(length(x0), tn);
    init = inte_rk8(tspan(1:n), x0, f);
    y(:, 1:n) = init;
    h = tspan(2)-tspan(1);
    for i = n+1:tn
        s = 0;
        for j = 1:n
            s = s+adamc(n,j)*f(0, y(:, i-(n+1-j)));
        end
        y(:, i) = y(:, i-1) + h*s;
    end
end

function y = pece(tspan, x0, f, n, adamc, adamc2)
    tn = length(tspan);
    y = zeros(length(x0), tn);
    init = inte_rk8(tspan(1:n), x0, f);
    y(:, 1:n) = init;
    h = tspan(2)-tspan(1);
    for i = n+1:tn
        s = 0;
        for j = 1:n
            s = s+adamc(n,j)*f(0, y(:, i-(n+1-j)));
        end
        y(:, i) = y(:, i-1) + h*s;
        
        s = 0;        
        for j = 1:n
            s = s+adamc2(n,j)*f(0, y(:, i-(n-j)));
        end
        y(:, i) = y(:, i-1) + h*s;
    end
end

function y = pe2ce(tspan, x0, f, n, adamc, adamc2)
    tn = length(tspan);
    y = zeros(length(x0), tn);
    init = inte_rk8(tspan(1:n), x0, f);
    y(:, 1:n) = init;
    h = tspan(2)-tspan(1);
    for i = n+1:tn
        s = 0;
        for j = 1:n
            s = s+adamc(n,j)*f(0, y(:, i-(n+1-j)));
        end
        y(:, i) = y(:, i-1) + h*s;
     
        for k = 1:3
            s = 0;    
            for j = 1:n
                s = s+adamc2(n,j)*f(0, y(:, i-(n-j)));
            end
            y(:, i) = y(:, i-1) + h*s;
        end

    end
end

function y = pemcme(tspan, x0, f, n, adamc, adamc2)
    tn = length(tspan);
    y = zeros(length(x0), tn);
    yB = zeros(length(x0), tn);
    yM = zeros(length(x0), tn);
    init = inte_rk8(tspan(1:n), x0, f);
    y(:, 1:n) = init;
    h = tspan(2)-tspan(1);
    cb = adamc2(n+1, n+1);
    cm = adamc2(n+1, 1);
    testflag = 0;
    for i = n+1:tn
        s = 0;
        for j = 1:n
            s = s+adamc(n,j)*f(0, y(:, i-(n+1-j)));
        end
        yB(:, i) = y(:, i-1) + h*s;
        y(:, i) = yB(:, i) + cb/(cb-cm)*(yM(:, i-1)-yB(:, i-1));

        s = 0;        
        for j = 1:n
            s = s+adamc2(n,j)*f(0, y(:, i-(n-j)));
        end
        yM(:, i) = y(:, i-1) + h*s;
        y(:, i) = yM(:, i) + cm/(cb-cm)*testflag*(yM(:, i)-yB(:, i));
        testflag = 1;
    end
end

function y = inte_rk8(tspan, x0, f)

    dt = tspan(2)-tspan(1);
    x = x0;
    data = [x0, zeros(length(x0),length(tspan)-1)];
    for i = 1:length(tspan)-1
        dx = RK(tspan(i+1), dt, x, f);
        x = x + dx*dt;
        data(:, i+1) = x;
    end
    y = data;
end

function dx = RK(t, h, x, f)
    [A, B, C]= rkcoedata;
    k = zeros(length(x), length(B));
    k(:,1) = f(t, x);
    for i = 2:length(B)
        k(:,i) = f(t+C(i)*h, x+sum(A(i,1:i-1).*k(:,1:i-1),2)*h);
    end
    dx = k*B;
end
function [A, B, C] = rkcoedata
A=[0 0 0 0 0 0 0 0 0 0 0 0 0
   2/27 0 0 0 0 0 0 0 0 0 0 0 0 
   1/36 1/12 0 0 0 0 0 0 0 0 0 0 0 
   1/24 0 1/8 0 0 0 0 0 0 0 0 0 0
   5/12 0 -25/16 25/16 0 0 0 0 0 0 0 0 0
   1/20 0 0 1/4 1/5 0 0 0 0 0 0 0 0
   -25/108 0 0 125/108 -65/27 125/54 0 0 0 0 0 0 0
   31/300 0 0 0 61/225 -2/9 13/900 0 0 0 0 0 0
   2 0 0 -53/6 704/45 -107/9 67/90 3 0 0 0 0 0
   -91/108 0 0 23/108 -976/135 311/54 -19/60 17/6 -1/12 0 0 0 0
   2383/4100 0 0 -341/164 4496/1025 -301/82 2133/4100 45/82 45/164 18/41 0 0 0
   3/205 0 0 0 0 -6/41 -3/205 -3/41 3/41 6/41 0 0 0
   -1777/4100 0 0 -341/164 4496/1025 -289/82 2193/4100 51/82 33/164 12/41 0 1 0];
B=[41/840;0;0;0;0;34/105;9/35;9/35;9/280;9/280;41/840];
C=[0;2/27;1/9;1/6;5/12;1/2;5/6;1/6;2/3;1/3;1;0;1];
end