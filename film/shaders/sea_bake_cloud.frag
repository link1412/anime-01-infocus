#version 410
// One cloud layer, raymarched once into a panorama: premultiplied radiance in rgb, transmittance in a.
// This is the expensive bit of the sky and it never changes, so it is baked and then only drifted.
// uRange is the window of the panorama: the part the camera can see is baked at about the pixel
// size of the frame, with noise octaves down to the width of a pixel at that distance.
in vec2 uv; out vec4 o;
#include "common.glsl"
#include "sea_atmo.glsl"
uniform sampler3D uNA,uNB,uNF;
uniform int uLayer;
uniform vec3 uLightDir;   // lighting direction, a touch below the true sun so the bases catch light
uniform vec2 uShells;     // altitude range km
uniform vec2 uOff;        // art-directable noise offset
uniform float uSigma,uAmb,uSunK,uPix;
uniform int uSteps;
uniform vec4 uRange;      // az0,az1,el0,el1; az0==az1 means the whole sky with v=sqrt(el)
uniform float uCold;

float sat(float x){ return clamp(x,0.,1.); }
float remap(float v,float a,float b,float c,float d){ return c+(v-a)/max(b-a,1e-4)*(d-c); }

float lodFor(float tile,float res,float fp){ return max(log2(fp*res/tile),0.); }

const mat2 R1=mat2(.8,.6,-.6,.8),R2=mat2(.28,.96,-.96,.28);
vec2 ROT(vec2 w,mat2 m){ return m*w; }

float sunGap=1.;
float rangeMask=1.;
float deckGap=1.;
float sunClear=1.;
float scrap=0.;
float crisp=1.;    // this side of the sky: the far side of the beach keeps its softer air

// the fractal with a cubic B-spline across its plane. Trilinear noise has corners on its lattice, and
// thresholding it for coverage draws those corners as the outline of a cloud.
float nfLevel(vec2 uv,float z,float l){
  float n=256./pow(2.,l);
  vec2 g=uv*n-.5,f=fract(g); g-=f;
  vec2 f2=f*f,f3=f2*f,c=1.-f;
  vec2 w0=c*c*c/6.,w1=(4.-6.*f2+3.*f3)/6.,w3=f3/6.,w2=1.-w0-w1-w3;
  vec2 a=w0+w1,b=w2+w3;
  vec2 p0=(g-.5+w1/a)/n,p1=(g+1.5+w3/b)/n;
  return a.y*(a.x*textureLod(uNF,vec3(p0,z),l).r+b.x*textureLod(uNF,vec3(p1.x,p0.y,z),l).r)
        +b.y*(a.x*textureLod(uNF,vec3(p0.x,p1.y,z),l).r+b.x*textureLod(uNF,vec3(p1,z),l).r);
}
float nfSmooth(vec2 uv,float z,float lod){
  float l=floor(lod),f=lod-l;
  float v=nfLevel(uv,z,l);
  if(f>=.02) v=mix(v,nfLevel(uv,z,l+1.),f);
  // the spline takes some of the noise's contrast with it: give that back
  return .5+1.15*(v-.5);
}

// p: km, world frame; fp: km per pixel across the ray. Returns extinction scale in [0,1].
float density(vec3 p,float alt,float fp){
  float hf=sat((alt-uShells.x)/(uShells.y-uShells.x));
  vec2 q=p.xz+uOff;
  if(uLayer==1){
    // the main deck: stratocumulus in loose rows of cells, with slots of open sky
    const float ca=.94,sa=.34;
    vec2 w=vec2(ca*q.x+sa*q.y,-sa*q.x+ca*q.y);
    // the air bends everything: no line in a cloud runs straight for long
    float lw=lodFor(37.,256.,fp)+2.;
    vec2 bend=vec2(textureLod(uNF,vec3(ROT(q,R1)/37.,.11),lw).r-.5,textureLod(uNF,vec3(ROT(q,R2)/29.,.73),lw).r-.5)*vec2(1.,.5);
    // what kind of weather is where: scales of plain fractal noise
    // and it changes with height, and leans with the wind aloft: a map of cover the same at every height
    // stands up as a prism, and a prism seen from below is a soft rectangle
    vec2 wc=w+bend*9.+(hf-.5)*vec2(1.6+9.*bend.y,.5);
    float big=nfSmooth(wc*vec2(1./260.,1./170.),.37+(alt-2.8)/40.,lodFor(170.,256.,fp));
    float mid=nfSmooth(wc*vec2(1./70.,1./45.),.61+(alt-2.8)/14.,lodFor(45.,256.,fp));
    float cov=sat(remap(big*.62+mid*.38,.37,.52,.0,1.))*rangeMask*deckGap*sunClear;
    if(cov<=0.) return 0.;
    // lumps, a few km across. The rows come from stretching the noise; its lattice is then turned away
    // from them, so that it lies along nothing the eye can follow
    w+=bend*16.;
    float n0=textureLod(uNA,vec3(ROT(w*vec2(1./19.,1./6.),R1),alt/6.),lodFor(6.,128.,fp)).r;
    float n1=nfSmooth(ROT(w*vec2(1./21.,1./8.),R2),alt/12.,lodFor(8.,256.,fp));
    float n2=textureLod(uNF,vec3(ROT(w,R2)/2.9,alt/2.7),lodFor(2.9,256.,fp)).r;
    float shape=n0*.3+n1*.54+n2*.16;
    float prof=smoothstep(0.,.16,hf)*smoothstep(1.,.5,hf);
    float d=sat(remap(shape*(.42+.58*prof),1.-cov*.74,.84,0.,1.));
    if(d<=0.) return 0.;
    // edges fray at every scale down to the pixel
    float e0=textureLod(uNF,vec3(ROT(w,R1)/.73,alt/.71),lodFor(.73,256.,fp)).r;
    vec4 a=textureLod(uNA,vec3(ROT(w,R2)/1.3,alt/1.1),lodFor(1.3,128.,fp));
    float ero=e0*.72+(a.g*.6+a.b*.4)*.28;
    // and how far they fray changes along the edge, or the thin rim becomes an outline
    float rimVar=textureLod(uNF,vec3(ROT(w,R2)/4.1,.3),lodFor(4.1,256.,fp)).r*.62+textureLod(uNF,vec3(ROT(w,R1)/1.17,.7),lodFor(1.17,256.,fp)).r*.38;
    rimVar=smoothstep(.3,.72,rimVar);
    d=sat(remap(d,ero*(.02+.86*rimVar),mix(.95,.62,crisp),0.,1.));
    // real edges are thin before they are white: optical depth grows slowly at first
    // but a cloud has an edge: below some density there is simply nothing there, least of all a pale scrap
    return mix(pow(d,1.35),pow(smoothstep(.05+scrap,.3+scrap,d),1.2)*d,crisp);
  }
  if(uLayer==0){
    // low dark rags under the deck, far out over the sea
    const float ca=.985,sa=-.17;
    vec2 w=vec2(ca*q.x+sa*q.y,-sa*q.x+ca*q.y);
    float big=textureLod(uNF,vec3(w*vec2(1./110.,1./33.),.13),lodFor(33.,256.,fp)).r;
    float mid=textureLod(uNF,vec3(w*vec2(1./27.,1./9.),.47),lodFor(9.,256.,fp)).r;
    float cov=sat(remap(big*.6+mid*.4,.52,.63,0.,.74));
    if(cov<=0.) return 0.;
    float n0=textureLod(uNF,vec3(ROT(w,R1)*vec2(1./4.3,1./2.1),alt/2.),lodFor(2.1,256.,fp)).r;
    float prof=smoothstep(0.,.3,hf)*smoothstep(1.,.35,hf);
    float d=sat(remap(n0*prof,1.-cov,.62,0.,1.));
    if(d<=0.) return 0.;
    float e0=textureLod(uNF,vec3(ROT(w,R2)/.51,alt/.4),lodFor(.51,256.,fp)).r;
    d=sat(remap(d,e0*.3,1.,0.,1.));
    return d*d*sunGap;
  }
  // the high veil over everything: thin cirrostratus, thin enough to see the sun through
  const float ca=.985,sa=.17;
  vec2 w=vec2(ca*q.x+sa*q.y,-sa*q.x+ca*q.y);
  float a=textureLod(uNF,vec3(w*vec2(1./300.,1./120.),.2),lodFor(120.,256.,fp)).r;
  float b=textureLod(uNF,vec3(ROT(w,R1)*vec2(1./34.,1./26.),.5),lodFor(26.,256.,fp)).r;
  float e=sat(remap(a,.34,.62,.62,1.));
  return e*sat(remap(b,.3,.7,.92,1.));
}

float lightTau(vec3 p,float fp){
  // exponentially spaced shadow rays towards the (low) light
  float tau=0.,dist=.05,prev=0.;
  const int N=10;
  float maxd=18.;
  for(int i=0;i<N;i++){
    float t=dist; float dt=t-prev;
    vec3 q=p+uLightDir*(.5*(t+prev));
    float alt=length(q)-RE;
    if(alt>uShells.y) break;
    if(alt>uShells.x) tau+=density(q,alt,max(fp,dt*.5))*dt;
    prev=t; dist*=pow(maxd/.05,1./float(N-1));
  }
  return tau*uSigma;
}

float phaseC(float c,float k){
  return mix(hg(c,-.28*k),hg(c,.86*k),.5);
}

vec3 grade(vec3 c){
  // cold air, a camera white-balanced for it: the low sun does not get to paint the sky orange
  float l=dot(c,vec3(.25,.65,.10));
  return mix(c,l*vec3(.965,.99,1.05),uCold);
}

void main(){
  vec3 ro=vec3(0.,RE+.006,0.);
  vec3 rd;
  if(uRange.x==uRange.y) rd=panoDir(uv);
  else{
    float az=mix(uRange.x,uRange.y,uv.x),el=mix(uRange.z,uRange.w,uv.y);
    float ce=cos(el);
    rd=vec3(sin(az)*ce,sin(el),cos(az)*ce);
  }
  rd=normalize(vec3(rd.x,max(rd.y,.0006),rd.z));
  vec2 lo=sphere(ro,rd,RE+uShells.x);
  vec2 hi=sphere(ro,rd,RE+uShells.y);
  float t0=max(lo.y,0.),t1=hi.y;
  if(t1<=t0||t0>620.){ o=vec4(0.,0.,0.,1.); return; }
  t1=min(t1,t0+260.);
  if(uLayer==2){
    float tH=.5*(t0+t1);
    vec3 pH=ro+rd*tH;
    // a thin plane seen edge on: the footprint along the ray decides, or the noise turns into scanlines
    float fp=tH*uPix*9./max(rd.y,.012);
    float d=density(pH,.5*(uShells.x+uShells.y),fp);
    float path=min(t1-t0,24.);
    float tau=uSigma*d*path;
    float Tc=exp(-tau);
    float cosS=dot(rd,uSunDir);
    vec3 sunC=exp(-extinction(odSun(pH,uSunDir)))*uE0*uWB;
    vec3 amb=vec3(.2,.2,.27)*uAmb;
    vec3 lum=sunC*(.34*hg(cosS,.8)+.66*hg(cosS,.2))*1.5+amb;
    vec3 Ta;
    vec3 Lf=scatter(ro,rd,tH,Ta,14);
    o=vec4(grade(Ta*lum*(1.-Tc)+(1.-Tc)*Lf),Tc);
    return;
  }
  int N=uSteps;
  float dt=(t1-t0)/float(N);
  uvec2 ju=uvec2(gl_FragCoord.xy)+uvec2(uLayer)*uvec2(1117u,2311u);
  ju=ju*1664525u+1013904223u; ju.x+=ju.y*1664525u; ju.y+=ju.x*1664525u; ju=ju^(ju>>16); ju.x+=ju.y*1664525u;
  float jit=float(ju.x>>8)/16777216.;
  float T=1.; vec3 L=vec3(0.);
  float tSum=0.,wSum=0.;
  float toSunA=length(rd-uSunDir);
  // a thin place in the low clouds where the sun comes through
  sunGap=mix(.2,1.,smoothstep(.02,.14,toSunA));
  // and the deck thins there too, down to bright frayed edges right around the sun
  deckGap=mix(.34,1.,smoothstep(.012,.17,toSunA));
  // no scraps around the sun: what the deck leaves there is its own solid edge
  sunClear=smoothstep(.1,.34,toSunA);
  scrap=.36*(1.-smoothstep(.25,.75,toSunA));
  crisp=1.-smoothstep(.5,.62,atan(rd.x,rd.z));
  sunClear=mix(1.,sunClear,crisp);
  float cosS=dot(rd,uLightDir);
  vec3 sunOD=odSun(ro+rd*(t0+.5*(t1-t0)),uSunDir);
  vec3 sunC=exp(-extinction(sunOD))*uE0*uWB;
  for(int i=0;i<N;i++){
    float t=t0+(float(i)+jit)*dt;
    vec3 p=ro+rd*t;
    float alt=length(p)-RE;
    // a pixel covers far more along the ray than across it: rows would alias against fine noise otherwise
    float fp=max(1.5*t*uPix/pow(max(rd.y,.04),.55),.004);
    // the deck stops somewhere over the sea, and the light comes in under its edge
    float dh=length(p.xz);
    float edge=26.+74.*textureLod(uNF,vec3(p.xz/230.,.5),2.).r;
    rangeMask=1.-smoothstep(edge,edge+22.,dh)*sat(.5+.5*dot(normalize(p.xz),normalize(uSunDir.xz))*2.5);
    float d=density(p,alt,fp);
    if(uLayer==0) d*=smoothstep(14.,34.,t)*(1.-smoothstep(40.,85.,t));
    if(d<=.002) continue;
    float tauL=lightTau(p,fp*2.);
    // the sun gets further into a cloud in some places than in others: a silver edge that comes and goes, never a line
    if(uLayer==1){
      vec2 qr=p.xz+uOff;
      float rr=textureLod(uNF,vec3(ROT(qr,R1)/17.,.17),lodFor(17.,256.,fp)).r*.55+textureLod(uNF,vec3(ROT(qr,R2)/6.1,.83),lodFor(6.1,256.,fp)).r*.45;
      tauL*=.5+1.2*smoothstep(.3,.7,rr);
    }
    float s=0.,a=1.,b=1.,k=1.;
    for(int o_=0;o_<5;o_++){
      s+=a*phaseC(cosS,k)*exp(-tauL*b);
      // most of what gets out of a cloud has bounced many times: inside is not so far below the rim
      a*=.6; b*=.37; k*=.58;
    }
    float hf=sat((alt-uShells.x)/(uShells.y-uShells.x));
    float toSun=.5+.5*dot(normalize(rd.xz),normalize(uSunDir.xz));
    vec3 ambTop=vec3(.15,.175,.24);
    vec3 ambLow=mix(vec3(.13,.13,.15),vec3(.3,.29,.3),toSun*toSun);
    vec3 amb=mix(ambLow,ambTop,smoothstep(.0,.9,hf))*uAmb;
    // light that has bounced around in the deck is not uniform: the underside is mottled
    if(uLayer==1){
      vec2 qa=p.xz+uOff;
      float mo=textureLod(uNF,vec3(qa/9.3,alt/1.7),lodFor(9.3,256.,fp)).r*.55+textureLod(uNF,vec3(qa.yx/1.9,alt/.6),lodFor(1.9,256.,fp)).r*.27+textureLod(uNF,vec3(qa*vec2(1./.61,1./.43),alt/.3),lodFor(.43,256.,fp)).r*.18;
      amb*=.12+1.9*smoothstep(.34,.66,mo);
    }
    float sigma=uSigma*d;
    vec3 lum=sunC*s*uSunK*(1.-exp(-sigma*.6))+amb;
    float dT=exp(-sigma*dt);
    L+=T*lum*(1.-dT);
    tSum+=t*T*(1.-dT); wSum+=T*(1.-dT);
    T*=dT;
    if(T<.004) break;
  }
  if(wSum<=1e-5){ o=vec4(0.,0.,0.,1.); return; }
  float tm=tSum/wSum;
  vec3 Ta;
  vec3 Lf=scatter(ro,rd,tm,Ta,14);
  o=vec4(grade(Ta*L+(1.-T)*Lf),T);
}
