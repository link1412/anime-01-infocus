#version 410
// Where each bit of the water surface was a moment ago: two sets of coordinates carried by the flow
// and restarted in turn, half a period apart. Whatever is drawn on the water through them travels
// exactly with it, and neither set is ever old enough to smear.
in vec2 uv; out vec4 o;
uniform sampler2D uS,uM;
uniform vec4 uGrid;
uniform ivec2 uN;
uniform float uDt;
uniform vec2 uKeep;
vec2 vel(ivec2 p){
  vec4 s=texelFetch(uS,clamp(p,ivec2(0),uN-1),0);
  return s.x>.0015?s.yz/s.x:vec2(0.);
}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec2 u=vel(p);
  float D=uGrid.y;
  float th=uGrid.x+(float(p.x)+.5)*D;
  float r=exp(uGrid.z+(float(p.y)+.5)*uGrid.w);
  vec2 nr=vec2(sin(th),cos(th)),nt=vec2(nr.y,-nr.x);
  vec2 back=vec2(dot(u,nt),dot(u,nr))/(r*D)*uDt;
  vec2 up=uv-back/vec2(uN);
  vec4 m=texture(uM,up);
  // where the flow has drawn the coordinates out too far they relax towards their surroundings: the
  // pattern still travels with the water, it just is not pulled into threads, which foam does not survive
  float cell=r*D;
  float k=clamp(.45/cell,1.,14.);
  vec2 e=vec2(k)/vec2(uN);
  vec4 mE=texture(uM,up+vec2(e.x,0.)),mW=texture(uM,up-vec2(e.x,0.)),mN=texture(uM,up+vec2(0.,e.y)),mS=texture(uM,up-vec2(0.,e.y));
  vec4 dx=mE-mW,dy=mN-mS;
  vec2 st=vec2(sqrt(dot(dx.xy,dx.xy)+dot(dy.xy,dy.xy)),sqrt(dot(dx.zw,dx.zw)+dot(dy.zw,dy.zw)))/(2.*k*cell);
  vec2 give=clamp(uDt*7.*max(st-.55,0.),0.,1.);
  vec4 mean=.25*(mE+mW+mN+mS);
  m=mix(m,mean,give.xxyy);
  m-=vec4(u,u)*uDt;
  o=m*uKeep.xxyy;
}
