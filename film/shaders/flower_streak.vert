#version 410
in vec2 p;
in vec4 iseg; in vec3 ibody;     // x0,y0,x1,y1 in clip units; half width, brightness, softness
uniform float uAspect;
out vec2 vS; out vec3 vB; out float vL;
void main(){
  vec2 a=iseg.xy,b=iseg.zw;
  vec2 d=b-a; float L=length(d*vec2(uAspect,1.));
  vec2 u=normalize(d*vec2(uAspect,1.));
  vec2 n=vec2(-u.y,u.x);
  float hw=ibody.x;
  vec2 c=(a+b)*.5*vec2(uAspect,1.);
  vec2 q=c+u*p.x*(L*.5+hw)+n*p.y*hw;
  gl_Position=vec4(q/vec2(uAspect,1.),0.,1.);
  vS=vec2(p.x*(L*.5+hw),p.y*hw); vB=ibody; vL=L*.5;
}
