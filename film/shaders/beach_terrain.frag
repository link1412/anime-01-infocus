#version 410
// The static sand at the scale of a pixel, baked once in the same log-polar frame as the water:
// wind ripples, drainage rills, swash marks, pebbles. No tile, nothing repeats.
in vec2 uv; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
#include "common.glsl"
uniform vec4 uTer;     // th0,th1,rho0,rho1
uniform float uTexel;  // rad / ln r per texel
uniform sampler2D uBed;
uniform vec4 uPol;
uniform ivec2 uNS;

float gn(vec2 p){ // smooth gradient-ish noise in -1..1
  return 2.*(vnoise(p)*.65+vnoise(p*2.03+7.1)*.35)-1.;
}
float fbmS(vec2 p,int oct){ float a=.5,s=0.; mat2 m=mat2(1.62,1.18,-1.18,1.62); for(int i=0;i<oct;i++){ s+=a*gn(p); p=m*p+vec2(3.1,1.7); a*=.52; } return s; }

float texm;  // metres per texel here
float fade(float lam){ return smoothstep(2.2,4.5,lam/texm); }

// wind ripples: sinuous crests that fork and die, patchy
float ripples(vec2 q,float mask){
  if(mask<=0.) return 0.;
  vec2 w=q+vec2(fbmS(q*.31,3),fbmS(q*.31+9.,3))*1.1+vec2(fbmS(q*2.1,2),fbmS(q*2.1+5.,2))*.12;
  float lam=.11+.05*gn(q*.13);
  float ph=(w.x*.32+w.y*.95)/lam;
  float s=sin(6.2832*ph+3.*gn(q*.7));
  s=s>0.?pow(s,.7):-pow(-s,1.4);   // sharp crest, soft trough
  // in patches, with bare sand between where the wind did not get at it
  float patchy=smoothstep(-.02,.46,fbmS(q*.27+3.3,3))*(.55+.45*smoothstep(-.3,.4,fbmS(q*1.3+7.,2)));
  return .00165*s*patchy*mask*fade(lam);
}
// the little channels the backwash cuts, running down the beach and joining up
float rills(vec2 q,vec2 down,float mask){
  if(mask<=0.) return 0.;
  vec2 ac=vec2(down.y,-down.x);
  vec2 s=vec2(dot(q,ac),dot(q,down));
  s.x+=fbmS(s*.5,3)*1.1+fbmS(s*2.3+4.,2)*.14;
  float a=1.-abs(gn(vec2(s.x*2.3,s.y*.16)));
  float b=1.-abs(gn(vec2(s.x*6.3+3.,s.y*.5+1.)));
  float c=1.-abs(gn(vec2(s.x*17.+9.,s.y*1.5+5.)));
  float dep=pow(a,5.)*.021*fade(.3)+pow(b,4.)*.0075*fade(.1)+pow(c,3.)*.002*fade(.04);
  float where=smoothstep(.1,.7,fbmS(q*vec2(.1,.08)+1.7,3)+.35);
  return -dep*where*mask;
}
// pebbles: grains of the basalt big enough to see
float pebbles(vec2 q,float dens,out float cover){
  cover=0.;
  if(dens<=0.) return 0.;
  float best=0.;
  for(int oc=0;oc<2;oc++){
    float sc=oc==0?.085:.034;
    if(texm*2.5>sc) continue;
    vec2 g=q/sc; vec2 c=floor(g);
    for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
      vec2 id=c+vec2(i,j);
      vec2 h=hash22(id+float(oc)*37.);
      float pr=hash21(id*1.7+3.);
      if(pr>dens) continue;
      vec2 ctr=id+h;
      float rad=(.22+.42*hash21(id+9.))*(oc==0?1.:.8);
      vec2 dd=g-ctr;
      float an=6.28*hash21(id+5.);
      float ca=cos(an),sa=sin(an);
      dd=vec2(ca*dd.x+sa*dd.y,(-sa*dd.x+ca*dd.y)*1.5);
      float d2=dot(dd,dd)/(rad*rad);
      d2*=1.+.45*sin(atan(dd.y,dd.x)*3.+an*5.)*hash21(id+2.)+.25*sin(atan(dd.y,dd.x)*5.+an*9.);   // no two the same, none round
      if(d2<1.){
        float hgt=sc*rad*.42*sqrt(1.-d2);
        if(hgt>best){ best=hgt; cover=1.; }
      }
    }
  }
  return best;
}
float heightAt(vec2 q,vec2 down,float wetz,float dry,float wash,out float pc){
  float hd=ripples(q,dry);
  hd+=rills(q,down,wetz);
  // fine lumps of the coarse sand
  hd+=.0022*fbmS(q*14.,3)*fade(.05)+.0009*gn(q*48.)*fade(.02)+.006*fbmS(q*2.1+1.,3)*fade(.3);
  // patches and strands of gravel, most where the swash leaves them
  float dens=smoothstep(.22,.7,fbmS(q*.33+8.,3))*.5+smoothstep(.3,.75,fbmS(q*1.7+2.,3))*.35;
  hd+=pebbles(q,dens*.075*wash,pc);
  return hd;
}
void main(){
  float th=mix(uTer.x,uTer.y,uv.x);
  float rho=mix(uTer.z,uTer.w,uv.y);
  float r=exp(rho);
  texm=r*uTexel;
  vec2 q=r*vec2(sin(th),cos(th));
  vec2 su=vec2((th-uPol.x)/(uPol.y*float(uNS.x)),(rho-uPol.z)/(uPol.w*float(uNS.y)));
  float z=texture(uBed,su).r;
  // the beach runs down towards the water here
  float e=r*uPol.y*2.;
  float za=texture(uBed,su+vec2(1.5/float(uNS.x),0.)).r,zb=texture(uBed,su-vec2(1.5/float(uNS.x),0.)).r;
  float zc=texture(uBed,su+vec2(0.,1.5/float(uNS.y))).r,zd=texture(uBed,su-vec2(0.,1.5/float(uNS.y))).r;
  vec2 nr=vec2(sin(th),cos(th)),nt=vec2(nr.y,-nr.x);
  vec2 grad=((za-zb)*nt+(zc-zd)*nr)/(3.*r*uPol.y);
  vec2 down=normalize(-grad+vec2(1e-3,-1e-4));
  float dry=smoothstep(.3,.62,z);
  float wetz=smoothstep(1.35,.9,z)*smoothstep(-.9,-.4,z);
  float pc;
  float e2=max(texm,.004);
  // the swash sorts its stones up to the berm and takes them from below it
  float wash=smoothstep(.45,.8,z);
  float h0=heightAt(q,down,wetz,dry,wash,pc);
  float pcx,pcz;
  float hx=heightAt(q+vec2(e2,0.),down,wetz,dry,wash,pcx);
  float hz=heightAt(q+vec2(0.,e2),down,wetz,dry,wash,pcz);
  vec2 sl=clamp(vec2(hx-h0,hz-h0)/e2,vec2(-1.5),vec2(1.5));
  // colour of the sand: almost black, sorted into faintly lighter and darker drifts
  float drift=.82+.2*fbmS(q*.17,4)+.1*fbmS(q*1.3+4.,3);
  // swash marks: thin arcs of sorted grains, following the contours of the beach
  float cz=z*11.+1.2*fbmS(q*.11,3)+.25*fbmS(q*1.1+3.,2);
  float mark=pow(1.-abs(2.*fract(cz)-1.),6.)*smoothstep(1.5,.9,z)*smoothstep(.0,.35,z)*smoothstep(-.1,.5,gn(q*vec2(.12,.05)+5.));
  float alb=drift*(1.+.5*mark)*(1.-.18*pc);
  o0=vec4(h0,sl,alb);
  o1=vec4(pc,mark,sl.x*sl.x,sl.y*sl.y);
}
