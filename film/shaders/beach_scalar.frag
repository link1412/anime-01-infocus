#version 410
// What the water carries and what it leaves behind, advected semi-Lagrangian on the polar grid:
// foam (r) and wetness of the sand (g).
in vec2 uv; out vec4 o;
uniform sampler2D uS,uB;
uniform vec4 uGrid;
uniform ivec2 uN;
uniform float uDt,uT;
const float G=9.81;
#include "common.glsl"
vec2 vel(ivec2 p){
  vec4 s=texelFetch(uS,clamp(p,ivec2(0),uN-1),0);
  return s.x>.0015?s.yz/s.x:vec2(0.);
}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 s=texelFetch(uS,p,0);
  float h=s.x;
  vec2 u=vel(p);
  float D=uGrid.y;
  float th=uGrid.x+(float(p.x)+.5)*D;
  float r=exp(uGrid.z+(float(p.y)+.5)*uGrid.w);
  vec2 nr=vec2(sin(th),cos(th)),nt=vec2(nr.y,-nr.x);
  float ur=dot(u,nr),ut=dot(u,nt);
  vec2 back=vec2(ut,ur)/(r*D)*uDt;
  vec4 b=texture(uB,uv-back/vec2(uN));
  // convergence and how fast the surface is rising, against the local wave speed
  vec2 ue=vel(p+ivec2(1,0)),uw=vel(p-ivec2(1,0)),un=vel(p+ivec2(0,1)),us=vel(p-ivec2(0,1));
  float dudr=dot(un-us,nr)/(2.*D);
  float dtdt=dot(ue-uw,nt)/(2.*D);
  float div=(dudr+ur+dtdt)/r;
  float c=sqrt(G*max(h,.0005));
  float rise=-div*h/c;
  // a wave breaks when its face rises faster than a fraction of the local wave speed (Kennedy et al.)
  float brk=smoothstep(.13,.42,rise)*smoothstep(.04,.22,h)*(1.-smoothstep(2.4,3.4,h));
  // a long wave does not go all at once: a section breaks while the shoulders beside it still stand
  vec2 qs=r*nr;
  float sec=vnoise(qs/vec2(19.,34.)+vec2(0.,uT*.09))*.65+vnoise(qs/vec2(6.,15.)+vec2(7.,-uT*.15))*.35;
  brk*=mix(1.,smoothstep(.3,.62,sec),smoothstep(.25,.8,h));
  float F=b.r;
  // foam rides the water; stranded on sand it collapses within seconds
  float tau=mix(6.5,4.5,smoothstep(.03,.6,h))*mix(.18,1.,smoothstep(.001,.012,h));
  F*=exp(-uDt/tau);
  // bubbles pile up where the flow converges: the bright line along the tip of a swash tongue
  float pile=max(-div,0.)*smoothstep(.0,.02,h)*(1.-smoothstep(.05,.14,h));
  F*=1.+min(pile*uDt*.5,.06);
  F=min(F+brk*11.*uDt,1.2);
  // the film drains into the sand, slowest low on the beach where the sand is saturated
  float W=b.g;
  float tw=mix(16.,90.,smoothstep(.9,.1,s.w));
  W*=exp(-uDt/tw);
  if(h>.002) W=1.;
  o=vec4(F,W,0.,1.);
}
