#version 410
in vec2 p; in vec2 ipos; in float isize,icell,icoc; in vec3 icol; in float ifade,icell2;
uniform float uAspect,uPxPerUnit;
out vec2 vP; flat out float vS,vCoc,vHalf,vCell,vCell2; flat out vec3 vC;
void main(){
  float half_=max(isize*.5,icoc)*1.04;
  vHalf=half_; vS=isize*.5; vCoc=icoc; vCell=icell; vCell2=icell2;
  vC=icol*ifade;
  vP=p*half_;
  vec2 v=ipos+p*half_;
  gl_Position=ifade>0.?vec4(v/vec2(uAspect,1.),0.,1.):vec4(3.,3.,3.,1.);
}
