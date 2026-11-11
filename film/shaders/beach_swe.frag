#version 410
// One stage of a finite-volume shallow-water step on the log-polar grid around the camera.
// Momentum stays in the world frame; every cell is a closed polygon, so still water stays still
// (hydrostatic reconstruction of Audusse et al.), and faces are second order away from the shore.
in vec2 uv; out vec4 o;
uniform sampler2D uS,uS0,uBC;
uniform vec4 uGrid;    // th0,dth,rho0,drho
uniform ivec2 uN;
uniform float uDt,uBCRow;
uniform int uStage;
const float G=9.81;
const float HV=.0015;   // below this depth water has no velocity of its own
const float VISC=.008;

struct C{ float h,u,v,z,e; };
C mk(vec4 s){
  C c; c.h=s.x; c.z=s.w; c.e=s.x+s.w;
  float k=s.x>HV?1./s.x:0.;
  // a thin film cannot run away from the solver
  c.u=clamp(s.y*k,-9.,9.); c.v=clamp(s.z*k,-9.,9.);
  return c;
}
C fetchC(ivec2 p){
  if(p.y>=uN.y){
    // offshore: the incident wave train, plus whatever is leaving (Flather)
    ivec2 q=ivec2(clamp(p.x,0,uN.x-1),uN.y-1);
    vec4 s=texelFetch(uS,q,0);
    vec4 bc=texture(uBC,vec2((float(q.x)+.5)/float(uN.x),uBCRow));
    C c=mk(s);
    float h0=-s.w;
    float cinv=sqrt(G/max(h0,1.));
    float th=uGrid.x+(float(q.x)+.5)*uGrid.y;
    vec2 nr=vec2(sin(th),cos(th));
    float eta=c.e;
    float dn=cinv*(eta-bc.x);
    c.e=bc.x; c.h=max(h0+bc.x,0.);
    c.u=bc.y+dn*nr.x; c.v=bc.z+dn*nr.y;
    return c;
  }
  p=clamp(p,ivec2(0),uN-1);
  return mk(texelFetch(uS,p,0));
}
float mm(float a,float b){
  // generalised minmod: sharper than minmod, still no new extrema
  const float T=1.45;
  if(a*b<=0.) return 0.;
  float s=sign(a);
  return s*min(min(abs(a),abs(b))*T,.5*abs(a+b));
}
struct S{ float e,h,u,v; };   // limited slopes
S slope(C a,C c,C b){
  S s;
  if(min(min(a.h,b.h),c.h)<.004){ s.e=0.;s.h=0.;s.u=0.;s.v=0.; return s; }
  s.e=mm(c.e-a.e,b.e-c.e);
  s.h=mm(c.h-a.h,b.h-c.h);
  s.u=mm(c.u-a.u,b.u-c.u);
  s.v=mm(c.v-a.v,b.v-c.v);
  return s;
}
// numerical flux through a face of unit normal n, from the side (a) of this cell to (b) of the neighbour
vec3 hll(float hL,float unL,float utL,float hR,float unR,float utR){
  if(hL<=0.&&hR<=0.) return vec3(0.);
  float cL=sqrt(G*hL),cR=sqrt(G*hR);
  float SL,SR;
  if(hL<=0.){ SL=unR-2.*cR; SR=unR+cR; }
  else if(hR<=0.){ SL=unL-cL; SR=unL+2.*cL; }
  else{
    float us=.5*(unL+unR)+cL-cR, cs=.5*(cL+cR)+.25*(unL-unR);
    SL=min(unL-cL,us-cs); SR=max(unR+cR,us+cs);
  }
  float f1L=hL*unL, f2L=hL*unL*unL+.5*G*hL*hL;
  float f1R=hR*unR, f2R=hR*unR*unR+.5*G*hR*hR;
  float f1,f2;
  if(SL>=0.){ f1=f1L; f2=f2L; }
  else if(SR<=0.){ f1=f1R; f2=f2R; }
  else{
    float id=1./max(SR-SL,1e-5);
    f1=(SR*f1L-SL*f1R+SL*SR*(hR-hL))*id;
    f2=(SR*f2L-SL*f2R+SL*SR*(hR*unR-hL*unL))*id;
  }
  return vec3(f1,f2,f1*(f1>0.?utL:utR));   // the contact keeps the shear sharp
}
vec2 wetLap(C a,C c){ return a.h>.004?a.h*vec2(a.u,a.v)-c.h*vec2(c.u,c.v):vec2(0.); }
vec3 dU=vec3(0.);
void face(C c,float e_f,float h_f,float u_f,float v_f,C nb,float e_n,float h_n,float u_n,float v_n,vec2 n,float L){
  float z_f=e_f-h_f, z_n=e_n-h_n;
  float zm=max(z_f,z_n);
  float hs=max(e_f-zm,0.), hr=max(e_n-zm,0.);
  vec2 t=vec2(-n.y,n.x);
  vec3 F=hll(hs,u_f*n.x+v_f*n.y,u_f*t.x+v_f*t.y,hr,u_n*n.x+v_n*n.y,u_n*t.x+v_n*t.y);
  // this side's pressure correction and the centred bed term keep still water still
  float p=.5*G*(h_f*h_f-hs*hs)-.5*G*(h_f+c.h)*(c.z-z_f);
  dU.x-=L*F.x;
  dU.yz-=L*(F.y*n+F.z*t+p*n);
}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 s=texelFetch(uS,p,0);
  vec4 s0=texelFetch(uS0,p,0);
  C c=mk(s);
  if(s.w>2.75){ o=s; return; }   // far above any wave: stay out of the solver
  C e=fetchC(p+ivec2(1,0)),w=fetchC(p-ivec2(1,0)),n=fetchC(p+ivec2(0,1)),q=fetchC(p-ivec2(0,1));
  float D=uGrid.y;
  float th=uGrid.x+(float(p.x)+.5)*D;
  float rho=uGrid.z+(float(p.y)+.5)*uGrid.w;
  float rn=exp(rho+.5*uGrid.w),rs=exp(rho-.5*uGrid.w);
  float A=.5*(rn*rn-rs*rs)*D;
  float chord=2.*sin(.5*D);
  vec2 nr=vec2(sin(th),cos(th));
  float tp=th+.5*D,tm=th-.5*D;
  vec2 nE=vec2(cos(tp),-sin(tp)),nW=-vec2(cos(tm),-sin(tm));
  S sx=slope(w,c,e),sy=slope(q,c,n);
  // the neighbours' slopes across the shared face
  if(c.h>0.||e.h>0.||w.h>0.||n.h>0.||q.h>0.){
    C ee=fetchC(p+ivec2(2,0)),ww=fetchC(p-ivec2(2,0)),nn=fetchC(p+ivec2(0,2)),qq=fetchC(p-ivec2(0,2));
    S se=slope(c,e,ee),sw=slope(ww,w,c),sn=slope(c,n,nn),sq=slope(qq,q,c);
    float Lr=rn-rs;
    face(c,c.e+.5*sx.e,c.h+.5*sx.h,c.u+.5*sx.u,c.v+.5*sx.v, e,e.e-.5*se.e,e.h-.5*se.h,e.u-.5*se.u,e.v-.5*se.v, nE,Lr);
    face(c,c.e-.5*sx.e,c.h-.5*sx.h,c.u-.5*sx.u,c.v-.5*sx.v, w,w.e+.5*sw.e,w.h+.5*sw.h,w.u+.5*sw.u,w.v+.5*sw.v, nW,Lr);
    face(c,c.e+.5*sy.e,c.h+.5*sy.h,c.u+.5*sy.u,c.v+.5*sy.v, n,n.e-.5*sn.e,n.h-.5*sn.h,n.u-.5*sn.u,n.v-.5*sn.v, nr,rn*chord);
    face(c,c.e-.5*sy.e,c.h-.5*sy.h,c.u-.5*sy.u,c.v-.5*sy.v, q,q.e+.5*sq.e,q.h+.5*sq.h,q.u+.5*sq.u,q.v+.5*sq.v, -nr,rs*chord);
  }
  vec3 r=s.xyz+uDt*dU/A;
  // out where the water is deep, unbroken waves are round: a long wave here has no dispersion to
  // hold it up, so an eddy viscosity keeps its front from becoming a shock until it is shoaling
  float visc=VISC*smoothstep(2.6,4.5,-s.w);     // by the bed, not the wave: a mask that follows the crests would eat water
  if(visc>0.){
    r.x+=visc*(e.e+w.e+n.e+q.e-4.*c.e);
    r.y+=visc*(e.h*e.u+w.h*w.u+n.h*n.u+q.h*q.u-4.*c.h*c.u);
    r.z+=visc*(e.h*e.v+w.h*w.v+n.h*n.v+q.h*q.v-4.*c.h*c.v);
  }
  // in the surf and the swash the turbulence carries momentum sideways, between wet cells only
  float dx=exp(rho)*D;
  float mix_=min(.05*uDt/(dx*dx),.2)*(1.-smoothstep(1.6,3.2,-s.w));
  if(mix_>0.&&c.h>HV) r.yz+=mix_*(wetLap(e,c)+wetLap(w,c)+wetLap(n,c)+wetLap(q,c));
  if(uStage==1) r=.5*(s0.xyz+r);
  float h=max(r.x,0.);
  vec2 m=r.yz;
  // infiltration into dry-ish coarse sand, and bed friction (Manning, semi-implicit)
  if(s.w>.15&&h<.05) h=max(h-uDt*.011,0.);
  if(h<HV*.5){ h=0.; m=vec2(0.); }
  else{
    float sp=length(m)/h;
    // the formula is for a boundary layer; a tongue a few cm thick on wet sand is not slowed without limit
    const float MANN=.018;
    float mann=MANN*(1.+.5*smoothstep(-.1,.35,s.w));   // coarse sand above the waterline holds a thin sheet back
    float cf=G*mann*mann*sp/pow(max(h,.1),1.333);
    m/=1.+uDt*cf;
    // cells shrink towards the camera: no front may cross more of one than the step can carry
    float cap=h*min(9.,max(.35*dx/uDt-sqrt(G*h),.3)); float mm_=length(m);
    if(mm_>cap) m*=cap/mm_;
  }
  o=vec4(h,m,s.w);
}
