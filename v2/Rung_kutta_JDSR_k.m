%子程序：解四阶龙格库塔法
function x=Rung_kutta_JDSR_k(a,b,h,x1)
x=zeros(1,length(x1));
for i=1:length(x1)-1
k1=h*(a*x(i)-b*x(i).^3+x1(i));
k2=h*(a*(x(i)+k1/2)-b*(x(i)+k1/2).^3+x1(i));
k3=h*(a*(x(i)+k2/2)-b*(x(i)+k2/2).^3+x1(i+1));
k4=h*(a*(x(i)+k3)-b*(x(i)+k3).^3+x1(i+1));
x(i+1)=x(i)+(1/6)*(k1+2*k2+2*k3+k4);
%  if abs(x(i+1))>5
%         x(i+1)=sign(x(i+1))*5;
%     else 
%         x(i+1)=x(i+1);
%     end
end
