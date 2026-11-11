#version 410
// Layers of baked ink slide at their own rates and are stacked back to front, each hiding
// what its marks reserve. Output: r=ink, g=vermilion, b=indigo (dilute far ink), a=region.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uFar,uHill,uMid,uWater,uNear,uFore,uSunT,uBoat,uMist,uSeal;
uniform vec2 uSealC; uniform float uSealS;
uniform float uAspect,uZoom,uT,uDmax,uLodBias,uHorizon,uPan;
uniform vec2 uZc;
uniform vec4 uPh;   // X phase of far,hill,mid,near
uniform vec4 uPer;  // period of far,hill,mid,near
uniform float uPhFore,uPerFore,uPerWater;
uniform vec2 uSunC; uniform vec4 uSunRect;
uniform vec2 uBoatC,uBoatSize;
uniform float uRG;     // far ridge growth, 0=a line on the horizon, 1=full
uniform vec4 uRev;     // per-layer reveal: mid, water, near, fore (1 means fully there)

// Things arrive through the mist as a soft front in the paper's own noise, washes before darks, the way a
// painter works. dirY>0: top first (a peak above a sea of mist). dirY<0: bottom first (things that grow).
float gate(float r,vec2 X,float D,float dirY){
  if(r>=1.) return 1.;
  if(r<=0.) return 0.;
  float ny=clamp((X.y+1.)*.5,0.,1.);
  float o=mix(ny,1.-ny,step(0.,dirY));
  o=o*.62+.26*texture(uMist,vec2(X.x/2.7+.31,X.y/1.7)).g+.12*texture(uMist,vec2(X.x/.5,X.y/.33)).r;
  o+=min(D,2.)*.05;
  return smoothstep(o-.14,o+.10,r*1.30-.08);
}
float dec(float x){ return x*x*uDmax; }
vec2 L(sampler2D s,float X,float P,float y,float b){
  vec4 t=texture(s,vec2(X/P,(y+1.)*.5),b);
  return vec2(dec(t.r),t.g);
}
// Mist is paper left alone, so it is a mask on the ink behind it.
float mist(vec2 v,float X,float s0,float s1,float y0,float y1,float lo,float hi,float seed){
  float n=texture(uMist,vec2((X+uT*s0)/8.+seed,(v.y+seed*.7)/2.)).r;
  float g=texture(uMist,vec2((X*1.7+uT*s1)/8.+seed*.3,(v.y*1.5-seed)/2.)).g;
  float wy=(n-.5)*.14;
  float band=smoothstep(y0,y0+(y1-y0)*.45,v.y+wy)*(1.-smoothstep(y1-(y1-y0)*.55,y1,v.y+wy));
  return smoothstep(lo,hi,band*(.45*n+.55*g+.42));
}

void main(){
  vec2 p=vec2((uv.x*2.-1.)*uAspect,uv.y*2.-1.);
  vec2 v=uZc*(1.-1./uZoom)+p/uZoom;
  float b=uLodBias;
  float ink=0.,red=0.,indigo=0.;
  float reg=0.;

  // ---- sun, and its dry vermilion lines on the water
  vec2 qs=(v-uSunC-uSunRect.xz)/vec2(uSunRect.y-uSunRect.x,uSunRect.w-uSunRect.z);
  // sampled outside any branch: implicit derivatives inside one pick garbage LODs at the sprite's edge
  vec4 s_=texture(uSunT,qs,b);
  float inS=step(0.,qs.x)*step(qs.x,1.)*step(0.,qs.y)*step(qs.y,1.);
  // the disc is there from the very first; only the lines on the water wait for the water
  red=dec(s_.r)*inS*mix(step(uHorizon,v.y),1.,uRev.y);
  // ---- far ranges: the ridge is stretched up from the horizon as it is revealed
  float Hz=uHorizon;
  float yFar=Hz+(v.y-Hz)/max(uRG,.012);
  vec2 f=L(uFar,v.x+uPh.x,uPer.x,yFar,b);
  red*=1.-f.y;
  // squeezed to almost nothing it is a dark horizon line
  indigo=f.x*(1.+2.2*(1.-smoothstep(.0,.8,uRG)));
  float yHill=Hz+(v.y-Hz)/max(smoothstep(.0,1.,uRG),.012);
  vec2 h=L(uHill,v.x+uPh.y,uPer.y,yHill,b);
  float reH=1.-h.y;
  float boost=1.+1.2*(1.-smoothstep(.2,.9,uRG));
  indigo=indigo*reH+h.x*.72*boost;
  ink=h.x*.28*boost;
  float m0=mist(v,v.x+uPh.y,.016,-.011,Hz-.10,Hz+.28,.60,.92,.13);
  indigo*=1.-m0*.8; ink*=1.-m0*.8;
  // mist is in front of the sun as well, so the hill's hard reserve dissolves with it
  red*=reH*(1.-m0*.8);
  // ---- middle distance
  // the great peak comes up out of the horizon too, the last of what the line turns into
  float gm=mix(.22,1.,uRev.x*uRev.x*(3.-2.*uRev.x));
  vec2 md=L(uMid,v.x+uPh.z,uPer.z,Hz+(v.y-Hz)/gm,b);
  { float k=gate(smoothstep(0.,.72,uRev.x),vec2(v.x+uPh.z,Hz+(v.y-Hz)/gm),md.x,-1.); md*=k; }
  float reM=1.-md.y;
  ink=ink*reM+md.x*.80;
  indigo=indigo*reM+md.x*.20;
  float m1=mist(v,v.x+uPh.z,.024,-.015,-.42,.22,.56,.90,.47);
  ink*=1.-m1; indigo*=1.-m1;
  red*=reM*(1.-m1*.8);
  // ---- water is a ground plane; its parallax grows with nearness
  float wk=clamp((uHorizon-v.y)/(uHorizon+1.),0.,1.);
  float wr=.30+.95*wk;
  float Xw=v.x+uPan*wr;
  float Pw=uPerWater;
  vec2 dxw=vec2(dFdx(v.x)/Pw,0.)*pow(2.,b), dyw=vec2(0.,dFdy(v.y)*.5)*pow(2.,b);
  vec4 w=textureGrad(uWater,vec2(Xw/Pw,(v.y+1.)*.5),dxw,dyw);
  float wd=dec(w.r)*step(v.y,uHorizon);
  ink+=wd*gate(uRev.y,vec2(Xw,v.y),0.,1.);
  // ---- boat
  vec2 qb=(v-uBoatC)/uBoatSize+.5;
  vec4 bt=texture(uBoat,qb,b*.5);
  float inB=step(0.,qb.x)*step(qb.x,1.)*step(0.,qb.y)*step(qb.y,1.)*gate(uRev.y,v,0.,1.);
  float bd=dec(bt.r)*inB, boatA=bt.g*inB;
  ink=ink*(1.-boatA)+bd; indigo*=1.-boatA; red*=1.-boatA;
  // ---- near
  vec2 n=L(uNear,v.x+uPh.w,uPer.w,v.y,b);
  { float k=gate(uRev.z,vec2(v.x+uPh.w,v.y),n.x,1.); n*=k; }
  float reN=1.-n.y;
  ink=ink*reN+n.x; indigo*=reN; red*=reN;
  float m2=mist(v,v.x+uPh.w,.034,-.020,-.46,.10,.58,.90,.81);
  ink*=1.-m2; indigo*=1.-m2; red*=1.-m2*.4;
  // ---- foreground
  vec2 g=L(uFore,v.x+uPhFore,uPerFore,v.y,b);
  { float k=gate(uRev.w,vec2(v.x+uPhFore,v.y),g.x,-1.); g*=k; }
  float reF=1.-g.y;
  ink=ink*reF+g.x; indigo*=reF; red*=reF;
  // ---- the painter's seal, on the sheet itself
  float dSeal=mod(v.x+uPh.w-uSealC.x+uPer.w*.5,uPer.w)-uPer.w*.5;
  vec2 qSeal=vec2(dSeal,v.y-uSealC.y)/uSealS+.5;
  float sealD=dec(texture(uSeal,qSeal,b).r)*step(0.,qSeal.x)*step(qSeal.x,1.)*step(0.,qSeal.y)*step(qSeal.y,1.)*smoothstep(.6,1.,uRev.z);
  red+=sealD*(1.-g.y);

  // what this place is, for the characters: <0 water, otherwise how much mist
  reg=max(max(m0*.8,m1),m2)*(1.-max(n.y,g.y));
  float land=max(max(n.y,g.y),max(md.y,h.y));
  if(v.y<uHorizon-.01) reg=mix(-1.,reg,clamp(max(n.y,g.y)+boatA,0.,1.));
  o=vec4(ink,red,indigo,reg);
}
